#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_TENSOR_CORES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_TENSOR_CORES_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdbool.h>
#include "card.cuh"   /* what a card's name is */

/* ══ ⭐ A TURING CARD WITHOUT TENSOR CORES — BY ITS NAME, A BEST GUESS ═════════════════════════════════════════════
 *
 * ⚖ 2026-11-09 *"if this holds for the 20 series then sm75 should belong to opencl"* · *"the issue is that 1660 had no
 * tensor cores... maybe that is the check that makes the discriminating factor"* · *"id rather keep the cuda for the high
 * performer than add optimizations to the lower end"*. `nvidia_cuda12`'s GEMVs and GEMMs run on the tensor cores
 * (`mma.sync`), and Turing's two smallest chips, TU116 and TU117, have none, though they are `sm_75` as the rest are.
 * `MEASURED` on vast.ai (measurements/2026-11-09_sm75_tensor_cores.md): on a GTX 1660 (TU116) the 4B `D4` decodes 12.2-12.4
 * tokens/s through `nvidia_cuda12` and 21.5 through `nvidia_opencl1` (25.8 with the read-only arenas), its 4-bit GEMVs
 * 22-30 GB/s against 75-102; on an RTX 2070 (TU106, tensor cores) 75.9-76.6 through `nvidia_cuda12`, the doors through
 * OpenCL 1.2 0.53× CUDA's at the median and the prompt's GEMM 1.3 TFLOP/s against 20-24.
 *
 * So a Turing card without them is named `sm_75:tc-` (`silicon_families__cards_present`, and `nvidia_cuda12` naming its
 * own devices): the flag after the colon changes no build target (`silicon_families__card_is` reads to the colon), and
 * `nvidia_cuda12` does not take it — `nvidia_opencl1` does, as it takes every NVIDIA card CUDA leaves.
 * ⛳ BY THE NAME THE DRIVER GIVES (`cuDeviceGetName`, `cudaDeviceProp::name`), matched anywhere in it — `NVIDIA GeForce GTX
 *   1660 SUPER`, `GeForce MX450`, `NVIDIA T1000 8GB`. `REASONED` from NVIDIA's specifications: the GTX 1630, 1650 and 1660
 *   (Ti, SUPER), the MX450 and MX550, the T400, T500, T550, T600, T1000 and T1200 and the Quadro T1000 and T2000 are TU116
 *   or TU117; the T4 (TU104) and every RTX are not. A card wrongly placed is a ticket: its name, and the list it moves to. */
static const char* const silicon_families__turing_without_tensor_cores[] = {
    "GTX 1630", "GTX 1650", "GTX 1660", "MX450", "MX550", "Quadro T1000", "Quadro T2000",
    " T400", " T500", " T550", " T600", " T1000", " T1200",
};

/* Whether `name` holds `part` anywhere. */
static inline bool silicon_families__name_holds(const char* name, const char* part) {
    if (name == 0 || part == 0) return false;
    for (; *name; ++name) {
        const char* a = name;
        const char* b = part;
        while (*b && *a == *b) { ++a; ++b; }
        if (*b == 0) return true;
    }
    return false;
}

/* Whether an NVIDIA card of compute capability `major`.`minor`, named `name` by its driver, is a Turing card without
 * tensor cores. */
static inline bool silicon_families__no_tensor_cores(int major, int minor, const char* name) {
    if (major != 7 || minor != 5) return false;
    for (unsigned i = 0; i < sizeof silicon_families__turing_without_tensor_cores / sizeof silicon_families__turing_without_tensor_cores[0]; ++i)
        if (silicon_families__name_holds(name, silicon_families__turing_without_tensor_cores[i])) return true;
    return false;
}

/* Whether `card`'s name carries `flag` after a colon (`sm_75:tc-`). */
static inline bool silicon_families__card_flag(const char* card, const char* flag) {
    if (card == 0 || flag == 0) return false;
    for (const char* c = card; *c; ++c) {
        if (*c != ':') continue;
        const char* a = c + 1;
        const char* b = flag;
        while (*b && *a == *b) { ++a; ++b; }
        if (*b == 0 && (*a == 0 || *a == ':')) return true;
    }
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_TENSOR_CORES_CUH */
