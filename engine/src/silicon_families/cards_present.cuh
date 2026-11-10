#ifndef SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH
#define SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdio.h>
#include <stdlib.h>                       /* getenv: the experimental lever */
#include <string.h>
#include "../operating_systems/os.cuh"   /* the system's cards, the NVIDIA driver's library: the system's */
#include "nvidia_tensor_cores.cuh"       /* a Turing card without tensor cores, by its name */
#if defined(__aarch64__)
#include <sys/auxv.h>                     /* an Arm processor's features, as the kernel reports them */
#include <asm/hwcap.h>
#endif
#include <stdint.h>

/* ══ ⭐⭐ THE CARDS THIS MACHINE HAS, BY NAME, BEFORE ANY FAMILY IS LOADED ══════════════════════════════
 *
 * ⚖ *"find the device name and check if it is included inside that family"*. The wheel asks this first,
 * offers each name to the families that include it (`choose.cuh`), and opens only the binaries of those. So the question is answered without a vendor runtime in the process: loading ROCm to learn that a
 * card needs a different ROCm would already be two ROCms in one process.
 *
 *     the system's  the GPUs the system itself names (`sys__os__cards`): on Linux the AMD kernel driver's
 *               topology (`gfx906`) — inside WSL2, which has none, AMD's GPUs by OpenCL's loader — on Windows
 *               OpenCL's loader — every GPU but NVIDIA's, by its OpenCL device name, an AMD card's its `gfx`.
 *     NVIDIA    the driver library, `libcuda.so.1`, which every NVIDIA box has and which is not a CUDA
 *               runtime: `cuInit`, then each device's compute capability — and its name, which marks a Turing
 *               card without tensor cores `sm_75:tc-` (▶ `nvidia_tensor_cores.cuh`). No library, no NVIDIA cards.
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

    /* ⚗ EXPERIMENTAL, ASKED FOR — `SILVANN_IGPU_CPU=1` (NN-142, ⚖ 2026-11-09 *"make it an experimental flag as igpu_cpu"*):
     *   Intel's GPUs where the system does not name them (Linux's), so an integrated GPU can be a model's CPU worker beside
     *   its card. Only on a machine whose system named no card: OpenCL's loader starts every vendor's part it lists, and
     *   ROCm's beside an AMD card the system named would be a second ROCm before its family opens (▶ `choose.cuh`). */
    const char* igpu = getenv("SILVANN_IGPU_CPU");
    if (n == 0u && most > 0u && igpu != 0 && strcmp(igpu, "1") == 0)
        n = sys__os__intel_gpus(names[0], SILICON_FAMILIES__CARD_NAME_BYTES, most);

    /* NVIDIA. ⛳ The driver's own four entry points, by name, so nothing here links against it. */
    const sys__os__library cuda = sys__os__library_open(SYS__OS__NVIDIA_DRIVER);
    if (cuda != 0) {
        typedef int (*init_fn)(unsigned int);
        typedef int (*count_fn)(int*);
        typedef int (*get_fn)(int*, int);
        typedef int (*attribute_fn)(int*, int, int);
        typedef int (*name_fn)(char*, int, int);
        const init_fn      init      = (init_fn)sys__os__library_find(cuda, "cuInit");
        const count_fn     count     = (count_fn)sys__os__library_find(cuda, "cuDeviceGetCount");
        const get_fn       get       = (get_fn)sys__os__library_find(cuda, "cuDeviceGet");
        const attribute_fn attribute = (attribute_fn)sys__os__library_find(cuda, "cuDeviceGetAttribute");
        const name_fn      named     = (name_fn)sys__os__library_find(cuda, "cuDeviceGetName");
        int devices = 0;
        if (init && count && get && attribute && init(0u) == 0 && count(&devices) == 0) {
            for (int i = 0; i < devices && n < most; ++i) {
                int device = 0, major = 0, minor = 0;
                /* 75 and 76 are CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR and _MINOR */
                if (get(&device, i) != 0 || attribute(&major, 75, device) != 0 || attribute(&minor, 76, device) != 0)
                    continue;
                char product[256] = {0};
                const bool bare = named && named(product, (int)sizeof product - 1, device) == 0
                               && silicon_families__no_tensor_cores(major, minor, product);
                snprintf(names[n++], SILICON_FAMILIES__CARD_NAME_BYTES, bare ? "sm_%d%d:tc-" : "sm_%d%d", major, minor);
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
#elif defined(__aarch64__)
    /* ⭐ AN ARM MACHINE'S: `arm_neon` where the processors have Armv8.2's half precision and dot product; `arm_gb10`
     *   where they are the DGX Spark's GB10 — Cortex-X925 (part 0xd85) and Cortex-A725 (0xd87) beside each other, as
     *   `/proc/cpuinfo` lists them — whose family has its own overrides (▶ arm_gb10/). */
    if (n < most && (getauxval(AT_HWCAP) & (HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_ASIMDDP)) == (HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_ASIMDDP)) {
        bool x925 = false, a725 = false;
        FILE* f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, "CPU part", 8) != 0) continue;
                if (strstr(line, "0xd85")) x925 = true;
                if (strstr(line, "0xd87")) a725 = true;
            }
            fclose(f);
        }
        snprintf(names[n++], SILICON_FAMILIES__CARD_NAME_BYTES, x925 && a725 ? "arm_gb10" : "arm_neon");
    }
#endif
    return n;
}

#endif /* SILVANN__SILICON_FAMILIES_CARDS_PRESENT_CUH */
