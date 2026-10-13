#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "sys.cuh"                      /* the contexts, the queues and the arenas */
#include "../../packages/nn/contracts/abi/gpu.cuh"   /* what is answered here */

/* ══ nn's REQUESTS, ANSWERED — khronos_opencl1_igpu ════════════════════════════════════════════════════════
 *
 * ⭐ THE KERNELS ARE OPENCL C 1.2, GENERATED FROM THE SAME ROWS AS EVERY OTHER FAMILY'S. The build
 *   (`scripts/opencl1_kernel_source.py`) flattens what a family's binary takes of nn — the compute bodies
 *   in `kernels/kernels.cuh` and the helpers before them — works out the space of every pointer in it, and
 *   writes one `__kernel` per row of nn's door list, running the body its `__RUNS` line names. The source
 *   rides inside this binary as a string and the runtime compiles it for the card the first time a kernel is
 *   launched. Every door runs nn's own body: the family has no overrides.
 * ⛳ EVERY POINTER ARGUMENT IS PASSED AS ITS ADDRESS, A 64-BIT NUMBER — an arena and an offset (`sys.cuh`) —
 *   and the door already hands the launch each argument's width, so the launch sets each argument by its
 *   bytes and never needs to know which were pointers.
 * ⛳ AND AFTER THEM, THE CARD'S ARENAS, one buffer argument each and null where a slot is empty: the kernel
 *   turns an address into its arena's pointer plus the offset. A kernel object keeps its arguments, so the
 *   arenas are set again only when the card's have changed since that kernel last had them.
 * ⛳ AN ARGUMENT INSIDE MAPPED RAM brings the program's copy to the card first (`sys.cuh`). */

/* nn's kernels as OpenCL C 1.2, one string, and how many kernels it holds. ⛳ DEFINED BY THE BUILD, NOT HERE:
 * `scripts/opencl1_kernel_source.py` writes them into a C file beside this binary's translation unit. */
extern "C" const char khronos_opencl1_igpu__kernel_source[];
extern "C" const unsigned khronos_opencl1_igpu__kernel_count;

typedef struct khronos_opencl1_igpu__kernel {
    const char* name;
    cl_kernel   kernel;
    uint64_t    generation;          /* the card's arenas as this kernel was last handed them; all ones: never */
} khronos_opencl1_igpu__kernel;

/* ⛳ ONE PROGRAM AND ONE SET OF KERNELS PER CARD, because each card has its own context (`sys.cuh`) and a
 *   program belongs to one. The table is made with the program. */
static cl_program               khronos_opencl1_igpu__zzprivate_program[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];
static bool                     khronos_opencl1_igpu__zzprivate_tried[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];
static khronos_opencl1_igpu__kernel* khronos_opencl1_igpu__zzprivate_kernels[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];
static uint32_t                 khronos_opencl1_igpu__zzprivate_kernel_count[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];

/* A card's program, built once, under the lock. A build that fails says why on stderr, once, and every
 * launch on that card after it does nothing — the verb's result then fails its own check. */
static inline bool khronos_opencl1_igpu__zzprivate_program_up(khronos_opencl1_igpu__state* s, uint32_t card) {
    if (khronos_opencl1_igpu__zzprivate_program[card]) return true;
    if (khronos_opencl1_igpu__zzprivate_tried[card]) return false;
    khronos_opencl1_igpu__zzprivate_tried[card] = true;
    const char* text = khronos_opencl1_igpu__kernel_source;
    cl_int err = CL_SUCCESS;
    cl_program p = clCreateProgramWithSource(s->context[card], 1u, &text, 0, &err);
    if (err != CL_SUCCESS) return false;
    if (clBuildProgram(p, 1u, &s->device[card], "-cl-std=CL1.2", 0, 0) != CL_SUCCESS) {
        size_t n = 0u;
        (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, 0, 0, &n);
        char* log = (char*)malloc(n + 1u);
        if (log) {
            (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, n, log, 0);
            log[n] = 0;
            fprintf(stderr, "khronos_opencl1_igpu: nn's kernels did not build:\n%s\n", log);
            free(log);
        }
        (void)clReleaseProgram(p);
        return false;
    }
    khronos_opencl1_igpu__zzprivate_kernels[card] =
        (khronos_opencl1_igpu__kernel*)calloc(khronos_opencl1_igpu__kernel_count, sizeof(khronos_opencl1_igpu__kernel));
    if (khronos_opencl1_igpu__zzprivate_kernels[card] == 0) { (void)clReleaseProgram(p); return false; }
    khronos_opencl1_igpu__zzprivate_program[card] = p;
    return true;
}

/* The kernel called `name` on `card`, made on first use; a kernel the program does not have is remembered
 * too, as a null one. */
static inline khronos_opencl1_igpu__kernel* khronos_opencl1_igpu__zzprivate_kernel(khronos_opencl1_igpu__state* s, uint32_t card, const char* name) {
    uint32_t* count = &khronos_opencl1_igpu__zzprivate_kernel_count[card];
    if (!khronos_opencl1_igpu__zzprivate_program_up(s, card)) return 0;
    khronos_opencl1_igpu__kernel* ks = khronos_opencl1_igpu__zzprivate_kernels[card];
    for (uint32_t i = 0u; i < *count; ++i)
        if (strcmp(ks[i].name, name) == 0) return ks[i].kernel ? &ks[i] : 0;
    if (*count == khronos_opencl1_igpu__kernel_count) return 0;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(khronos_opencl1_igpu__zzprivate_program[card], name, &err);
    if (err != CL_SUCCESS) k = 0;
    ks[*count].name = name; ks[*count].kernel = k; ks[*count].generation = ~0ull;
    return k ? &ks[(*count)++] : (++*count, (khronos_opencl1_igpu__kernel*)0);
}

/* Start the kernel called `name` on `blocks` blocks of `threads` threads. `kernel` is null: this family
 * compiles no kernel of its own, so there is no address to take.
 * ⛳ UNDER THE LOCK, because a kernel's arguments belong to the kernel object and not to the launch: two
 *   threads setting them at once would launch each other's. */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    (void)kernel;
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    cl_command_queue q = khronos_opencl1_igpu__zzpackage_queue();
    if (s == 0 || q == 0) return;
    pthread_mutex_lock(&khronos_opencl1_igpu__zzpackage_lock);
    const uint32_t card = khronos_opencl1_igpu__bound_device();
    khronos_opencl1_igpu__kernel* k = card < s->count ? khronos_opencl1_igpu__zzprivate_kernel(s, card, name) : 0;
    bool ok = k != 0;
    for (uint32_t i = 0u; ok && i < count; ++i) ok = clSetKernelArg(k->kernel, i, sizes[i], args[i]) == CL_SUCCESS;
    if (ok && k->generation != s->generation[card]) {
        for (uint32_t a = 0u; ok && a < KHRONOS_OPENCL1_IGPU__ARENAS; ++a)
            ok = clSetKernelArg(k->kernel, count + a, sizeof(cl_mem), &s->arena[card][a].buffer) == CL_SUCCESS;
        if (ok) k->generation = s->generation[card];
    }
    if (ok) ok = khronos_opencl1_igpu__zzpackage_launching(s, q, args, sizes, count);
    if (ok) {
        const size_t global = (size_t)blocks * (size_t)threads, local = (size_t)threads;
        ok = clEnqueueNDRangeKernel(q, k->kernel, 1u, 0, &global, &local, 0u, 0, 0) == CL_SUCCESS;
    }
    pthread_mutex_unlock(&khronos_opencl1_igpu__zzpackage_lock);
    if (!ok) fprintf(stderr, "khronos_opencl1_igpu: %s did not launch\n", name);
}

/* ⛳ THE ARITHMETIC IS THE KERNELS' OWN, written into their source by the build (OpenCL's `exp`, `sqrt`,
 *   `log`, `cos` and `sin`, the precise ones, and the lane functions). These host versions exist because this binary's
 *   host side includes nn's bodies too; nothing on the host calls them. */
static __device__ inline float nn__silicon__expf(float x)  { return expf(x); }
static __device__ inline float nn__silicon__sqrtf(float x) { return sqrtf(x); }
static __device__ inline float nn__silicon__logf(float x)  { return logf(x); }
static __device__ inline float nn__silicon__cosf(float x)  { return cosf(x); }
static __device__ inline float nn__silicon__sinf(float x)  { return sinf(x); }
/* The same for the lanes, the blocks and the combine step: on the card they are `get_local_id(0)`,
 * `get_local_size(0)`, `get_group_id(0)`, `get_num_groups(0)`, a fold through the block's shared memory and
 * `vload_half`, written into the kernels' source by the build; on the host side of this binary a body is
 * one lane of one. */
static __device__ inline uint32_t nn__silicon__lane(void)  { return 0u; }
static __device__ inline uint32_t nn__silicon__lanes(void) { return 1u; }
static __device__ inline uint32_t nn__silicon__block(void)  { return 0u; }
static __device__ inline uint32_t nn__silicon__blocks(void) { return 1u; }
static __device__ inline float nn__silicon__lanes_sum(float part) { return part; }
static __device__ inline void nn__silicon__lanes_sums(float* parts, uint32_t n) { (void)parts; (void)n; }
static __device__ inline float nn__silicon__lanes_max(float part) { return part; }
static __device__ inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {1u, 4u}; return t; }
static __device__ inline float nn__silicon__half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16, rest = (uint32_t)h & 0x7FFFu;
    union { uint32_t u; float f; } v;
    if (rest >= 0x7C00u) { v.u = sign | 0x7F800000u | ((rest & 0x3FFu) << 13); }
    else { v.u = rest << 13; v.f *= 0x1p112f; v.u |= sign; }
    return v.f;
}
/* Two halves by two halves onto a float, in the order a sum over the columns has. */
static __device__ inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c) {
    return (c + nn__silicon__half_to_float((uint16_t)(a & 0xffffu)) * nn__silicon__half_to_float((uint16_t)(b & 0xffffu)))
         + nn__silicon__half_to_float((uint16_t)(a >> 16)) * nn__silicon__half_to_float((uint16_t)(b >> 16));
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_NN_CUH */
