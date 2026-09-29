#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "sys.cuh"                      /* the context, the queues and the live list */
#include "../../packages/nn/contracts/abi/gpu.cuh"   /* what is answered here */

/* ══ nn's REQUESTS, ANSWERED — khronos_opencl2 ════════════════════════════════════════════════════════
 *
 * ⭐ THE KERNELS ARE OPENCL C, GENERATED FROM THE SAME ROWS AS EVERY OTHER FAMILY'S. The build
 *   (`scripts/opencl_kernel_source.py`) flattens what a family's binary takes of nn — the compute bodies
 *   in `kernels/kernels.cuh` and the helpers before them — and writes one `__kernel` per row of nn's door
 *   list, running the body its `__RUNS` line names. The source rides inside this binary as a string and
 *   the runtime compiles it for the card the first time a kernel is launched.
 * ⛳ EVERY POINTER ARGUMENT IS PASSED AS ITS ADDRESS, A 64-BIT NUMBER, and the kernel turns it back into a
 *   pointer: shared virtual memory makes it the same address on both sides, and the door already hands
 *   the launch each argument's width. So the launch sets each argument by its bytes and never needs to
 *   know which were pointers.
 * ⛳ AND THE CARD IS TOLD EVERY LIVE ALLOCATION IT OWNS, since an argument may point anywhere inside one —
 *   a coarse-grained allocation is only guaranteed present for a kernel that was told it may use it. */

/* nn's kernels as OpenCL C, one string. ⛳ DEFINED BY THE BUILD, NOT HERE: `scripts/opencl_kernel_source.py`
 * writes it into a C file beside this binary's translation unit, and the two are compiled together — so
 * nothing in the tree names a file outside it. */
extern "C" const char khronos_opencl2__kernel_source[];

typedef struct khronos_opencl2__kernel {
    const char* name;
    cl_kernel   kernel;
} khronos_opencl2__kernel;

#define KHRONOS_OPENCL2__KERNELS_MAX  64u
/* ⛳ ONE PROGRAM AND ONE SET OF KERNELS PER CARD, because each card has its own context (`sys.cuh`) and a
 *   program belongs to one. */
static cl_program              khronos_opencl2__zzprivate_program[KHRONOS_OPENCL2__DEVICES_MAX];
static bool                    khronos_opencl2__zzprivate_tried[KHRONOS_OPENCL2__DEVICES_MAX];
static khronos_opencl2__kernel khronos_opencl2__zzprivate_kernels[KHRONOS_OPENCL2__DEVICES_MAX][KHRONOS_OPENCL2__KERNELS_MAX];
static uint32_t                khronos_opencl2__zzprivate_kernel_count[KHRONOS_OPENCL2__DEVICES_MAX];

/* A card's program, built once, under the lock. A build that fails says why on stderr, once, and every
 * launch on that card after it does nothing — the verb's result then fails its own check. */
static inline bool khronos_opencl2__zzprivate_program_up(khronos_opencl2__state* s, uint32_t card) {
    if (khronos_opencl2__zzprivate_program[card]) return true;
    if (khronos_opencl2__zzprivate_tried[card]) return false;
    khronos_opencl2__zzprivate_tried[card] = true;
    const char* text = khronos_opencl2__kernel_source;
    cl_int err = CL_SUCCESS;
    cl_program p = clCreateProgramWithSource(s->context[card], 1u, &text, 0, &err);
    if (err != CL_SUCCESS) return false;
    if (clBuildProgram(p, 1u, &s->device[card], "-cl-std=CL2.0", 0, 0) != CL_SUCCESS) {
        size_t n = 0u;
        (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, 0, 0, &n);
        char* log = (char*)malloc(n + 1u);
        if (log) {
            (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, n, log, 0);
            log[n] = 0;
            fprintf(stderr, "khronos_opencl2: nn's kernels did not build:\n%s\n", log);
            free(log);
        }
        (void)clReleaseProgram(p);
        return false;
    }
    khronos_opencl2__zzprivate_program[card] = p;
    return true;
}

/* The kernel called `name` on `card`, made on first use. */
static inline cl_kernel khronos_opencl2__zzprivate_kernel(khronos_opencl2__state* s, uint32_t card, const char* name) {
    khronos_opencl2__kernel* ks = khronos_opencl2__zzprivate_kernels[card];
    uint32_t* count = &khronos_opencl2__zzprivate_kernel_count[card];
    for (uint32_t i = 0u; i < *count; ++i)
        if (strcmp(ks[i].name, name) == 0) return ks[i].kernel;
    if (!khronos_opencl2__zzprivate_program_up(s, card) || *count == KHRONOS_OPENCL2__KERNELS_MAX) return 0;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(khronos_opencl2__zzprivate_program[card], name, &err);
    if (err != CL_SUCCESS) return 0;
    ks[*count].name = name; ks[*count].kernel = k; ++*count;
    return k;
}

/* The bound card's live allocations, gathered for a launch — the ones its kernels may be handed. Under the
 * lock, into a list kept for the purpose. */
static void**  khronos_opencl2__zzprivate_named = 0;
static size_t  khronos_opencl2__zzprivate_named_room = 0u;
static inline size_t khronos_opencl2__zzprivate_name_card(khronos_opencl2__state* s, uint32_t card) {
    if (khronos_opencl2__zzprivate_named_room < s->live_count) {
        void** grown = (void**)realloc(khronos_opencl2__zzprivate_named, s->live_count * sizeof(void*));
        if (!grown) return 0u;
        khronos_opencl2__zzprivate_named = grown; khronos_opencl2__zzprivate_named_room = s->live_count;
    }
    size_t n = 0u;
    for (size_t i = 0u; i < s->live_count; ++i)
        if (s->live[i].card == card) khronos_opencl2__zzprivate_named[n++] = s->live[i].at;
    return n;
}

/* Start the kernel called `name` on `blocks` blocks of `threads` threads. `kernel` is null: this family
 * compiles no kernel of its own, so there is no address to take.
 * ⛳ UNDER THE LOCK, because a kernel's arguments belong to the kernel object and not to the launch: two
 *   threads setting them at once would launch each other's. */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    (void)kernel;
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    cl_command_queue q = khronos_opencl2__zzpackage_queue();
    if (s == 0 || q == 0) return;
    pthread_mutex_lock(&s->lock);
    const uint32_t card = khronos_opencl2__bound_device();
    cl_kernel k = card < s->count ? khronos_opencl2__zzprivate_kernel(s, card, name) : 0;
    bool ok = k != 0;
    for (uint32_t i = 0u; ok && i < count; ++i) ok = clSetKernelArg(k, i, sizes[i], args[i]) == CL_SUCCESS;
    const size_t named = ok ? khronos_opencl2__zzprivate_name_card(s, card) : 0u;
    if (ok && named)
        ok = clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_SVM_PTRS, named * sizeof(void*), khronos_opencl2__zzprivate_named) == CL_SUCCESS;
    if (ok) {
        const size_t global = (size_t)blocks * (size_t)threads, local = (size_t)threads;
        ok = clEnqueueNDRangeKernel(q, k, 1u, 0, &global, &local, 0u, 0, 0) == CL_SUCCESS;
    }
    pthread_mutex_unlock(&s->lock);
    if (!ok) fprintf(stderr, "khronos_opencl2: %s did not launch\n", name);
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
 * `get_local_size(0)`, `get_group_id(0)`, `get_num_groups(0)`, `work_group_reduce_add` / `_max` and
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

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_NN_CUH */
