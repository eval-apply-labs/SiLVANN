#ifndef SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH
#define SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdint.h>

/* ══ ⭐⭐ THE CARDS THIS MACHINE HAS, BY NAME, BEFORE ANY FAMILY IS LOADED ══════════════════════════════
 *
 * ⚖ *"find the device name and check if it is included inside that family"*. The wheel asks this first,
 * offers each name to the families that include it (`choose.cuh`), and opens only the binaries of those. So the question is answered without a vendor runtime in the process: loading ROCm to learn that a
 * card needs a different ROCm would already be two ROCms in one process.
 *
 *     AMD       the kernel driver's topology, `/sys/class/kfd/kfd/topology/nodes/<n>/properties`: every
 *               GPU node carries `gfx_target_version` as a number, major·10000 + minor·100 + stepping,
 *               and its name is `gfx` then the major in decimal and the other two in hex — 90006 is
 *               `gfx906`, 90010 `gfx90a`, 110000 `gfx1100`. A CPU node carries 0 and is skipped.
 *     NVIDIA    the driver library, `libcuda.so.1`, which every NVIDIA box has and which is not a CUDA
 *               runtime: `cuInit`, then each device's compute capability. No library, no NVIDIA cards.
 *
 * ⛳ ONLY THE PROGRAM'S SIDE CALLS THIS — the wheel, on Linux. It is a question about the machine and not
 *   about the silicon, so it lives beside the roster and in no family.
 * ⚠ THE ORDER IS THE DRIVERS' — kfd's node order and CUDA's device order — and nothing here promises it
 *   matches the order a family numbers its devices in. A family counts its own devices from its own
 *   runtime (`device_count`); this answers only which families are needed at all. */

#define SILICON_FAMILIES__CARD_NAME_BYTES  32u

/* The cards present, up to `most`, each name written into `names[i]`. Answers how many were written. */
static inline uint32_t silicon_families__cards_present(char (*names)[SILICON_FAMILIES__CARD_NAME_BYTES],
                                                       uint32_t most) {
    uint32_t n = 0u;

    /* AMD. ⛳ The nodes are read in numeric order, so the answer is the same on every read. */
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
        snprintf(names[n++], SILICON_FAMILIES__CARD_NAME_BYTES, "gfx%llu%llx%llx",
                 version / 10000ull, (version / 100ull) % 100ull, version % 100ull);
    }

    /* NVIDIA. ⛳ The driver's own four entry points, by name, so nothing here links against it. */
    void* cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (cuda != 0) {
        typedef int (*init_fn)(unsigned int);
        typedef int (*count_fn)(int*);
        typedef int (*get_fn)(int*, int);
        typedef int (*attribute_fn)(int*, int, int);
        const init_fn      init      = (init_fn)dlsym(cuda, "cuInit");
        const count_fn     count     = (count_fn)dlsym(cuda, "cuDeviceGetCount");
        const get_fn       get       = (get_fn)dlsym(cuda, "cuDeviceGet");
        const attribute_fn attribute = (attribute_fn)dlsym(cuda, "cuDeviceGetAttribute");
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
