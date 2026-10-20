#ifndef SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH
#define SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdio.h>
#include <string.h>
#include "../operating_systems/os.cuh"   /* the system's cards, the NVIDIA driver's library: the system's */
#include <stdint.h>

/* ══ ⭐⭐ THE CARDS THIS MACHINE HAS, BY NAME, BEFORE ANY FAMILY IS LOADED ══════════════════════════════
 *
 * ⚖ *"find the device name and check if it is included inside that family"*. The wheel asks this first,
 * offers each name to the families that include it (`choose.cuh`), and opens only the binaries of those. So the question is answered without a vendor runtime in the process: loading ROCm to learn that a
 * card needs a different ROCm would already be two ROCms in one process.
 *
 *     the system's  the GPUs the system itself names (`sys__os__cards`): on Linux the AMD kernel driver's
 *               topology (`gfx906`), on Windows OpenCL's loader — every GPU but NVIDIA's, by its OpenCL
 *               device name, an AMD card's its `gfx`.
 *     NVIDIA    the driver library, `libcuda.so.1`, which every NVIDIA box has and which is not a CUDA
 *               runtime: `cuInit`, then each device's compute capability. No library, no NVIDIA cards.
 *
 * ⛳ ONLY THE PROGRAM'S SIDE CALLS THIS — the wheel. It is a question about the machine and not
 *   about the silicon, so it lives beside the roster and in no family.
 * ⚠ THE ORDER IS THE DRIVERS' — kfd's node order and CUDA's device order — and nothing here promises it
 *   matches the order a family numbers its devices in. A family counts its own devices from its own
 *   runtime (`device_count`); this answers only which families are needed at all. */

#define SILICON_FAMILIES__CARD_NAME_BYTES  32u

/* The cards present, up to `most`, each name written into `names[i]`. Answers how many were written. */
static inline uint32_t silicon_families__cards_present(char (*names)[SILICON_FAMILIES__CARD_NAME_BYTES],
                                                       uint32_t most) {
    /* The system's own. */
    uint32_t n = most == 0u ? 0u : sys__os__cards(names[0], SILICON_FAMILIES__CARD_NAME_BYTES, most);

    /* NVIDIA. ⛳ The driver's own four entry points, by name, so nothing here links against it. */
    const sys__os__library cuda = sys__os__library_open(SYS__OS__NVIDIA_DRIVER);
    if (cuda != 0) {
        typedef int (*init_fn)(unsigned int);
        typedef int (*count_fn)(int*);
        typedef int (*get_fn)(int*, int);
        typedef int (*attribute_fn)(int*, int, int);
        const init_fn      init      = (init_fn)sys__os__library_find(cuda, "cuInit");
        const count_fn     count     = (count_fn)sys__os__library_find(cuda, "cuDeviceGetCount");
        const get_fn       get       = (get_fn)sys__os__library_find(cuda, "cuDeviceGet");
        const attribute_fn attribute = (attribute_fn)sys__os__library_find(cuda, "cuDeviceGetAttribute");
        int devices = 0;
        if (init && count && get && attribute && init(0u) == 0 && count(&devices) == 0) {
            for (int i = 0; i < devices && n < most; ++i) {
                int device = 0, major = 0, minor = 0;
                /* 75 and 76 are CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR */
                if (get(&device, i) != 0 || attribute(&major, 75, device) != 0 || attribute(&minor, 76, device) != 0)
                    continue;
                snprintf(names[n++], SILICON_FAMILIES__CARD_NAME_BYTES, "sm_%d%d", major, minor);
            }
        }
        /* ⛳ NOT CLOSED: the driver stays initialised for the process, and a family loaded next uses it. */
    }

    /* THE PROCESSORS, when they can run `x86_avx2`'s bodies — one card for the machine, and LAST, so a
     * machine with cards still stands up on its first card by default. */
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (n < most && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c"))
        snprintf(names[n++], SILICON_FAMILIES__CARD_NAME_BYTES, "x86_avx2");
#endif
    return n;
}

#endif /* SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH */
