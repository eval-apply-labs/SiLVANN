#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_NN_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "sys.cuh"                      /* the context, the queues and the live list */
#include "../../packages/nn/contracts/abi/gpu.cuh"   /* what is answered here */

/* ══ nn's REQUESTS, ANSWERED — khronos_opencl2_base ════════════════════════════════════════════════════════
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

/* nn's kernels as OpenCL C: the bodies every kernel's program begins with, and each kernel's name and own text — nn's
 * `__kernel` for a row, or a family override with its section's helpers and guard. ⛳ DEFINED BY THE BUILD, NOT HERE:
 * `scripts/opencl_kernel_source.py` writes them into a C file beside this binary's translation unit, and the two are
 * compiled together — so nothing in the tree names a file outside it. */
extern "C" const char khronos_opencl2_base__kernel_bodies[];
extern "C" const char* const khronos_opencl2_base__kernel_names[];
extern "C" const char* const khronos_opencl2_base__kernel_texts[];

/* ⭐ AND HOW MANY KERNELS THERE ARE — every package's with doors, as the build wrote them — which sizes each card's table.
 *   ⛳ THE BUILD COUNTS THEM BECAUSE ONLY IT KNOWS: it writes a kernel for each door row of each package that has doors,
 *   nn's today and a model package's the day one has a fused body. A written 64 here once stood while nn alone grew to
 *   80, and a kernel past the 64th was never made: its launch did nothing. */
extern "C" const unsigned khronos_opencl2_base__kernel_count;

/* ⭐⭐ A PROGRAM A KERNEL, BUILT THE FIRST TIME IT IS ASKED FOR — the bodies and that kernel's own text. A device's compiler
 *   that refuses one kernel then costs that kernel's door alone, which nn's host body answers (▶ the launch); one program
 *   of every kernel made a single refusal every door's. `MEASURED` Qualcomm's Adreno 660: the whole program refused,
 *   `CL_OUT_OF_HOST_MEMORY`, every launch on the card doing nothing; most of nn's kernels built alone in 1-2 s, three
 *   refused (measurements/2026-11-11_nn184_adreno_razer_edge.md). A kernel the program has not — an override whose guard
 *   the device's compiler leaves out, as where it has no sub-groups — is a kernel of none, as it was. */
typedef enum { KHRONOS_OPENCL2_BASE__UNTRIED = 0, KHRONOS_OPENCL2_BASE__BUILT, KHRONOS_OPENCL2_BASE__NONE } khronos_opencl2_base__made;
typedef struct khronos_opencl2_base__kernel {
    cl_program program;
    cl_kernel  kernel;
    khronos_opencl2_base__made made;
} khronos_opencl2_base__kernel;
static khronos_opencl2_base__kernel* khronos_opencl2_base__zzprivate_kernels[KHRONOS_OPENCL2_BASE__DEVICES_MAX];


/* ⛔⛔ KERNELS A DEVICE'S COMPILER IS KNOWN TO BE TAKEN DOWN BY, NEVER BUILT ON IT: answered as refused, so their doors are
 *   the overrides' or nn's other kernels or nn's host body. Qualcomm's compiler runs in the calling process and spends its
 *   memory there: `MEASURED` on the Razer Edge's Adreno 660 (measurements/2026-11-11_nn184_adreno_razer_edge.md), the
 *   phone rebooted itself while it built nn's grouped GEMV, and it took the family's GEMM tile past 860 MB in 14 s, built
 *   alone; nn's int8 expert GEMM past 720 MB in 30 s, still growing when it was stopped
 *   (measurements/2026-11-11_nn187_adreno_family_razer_edge.md). Their kin — the same body over other widths and forms —
 *   are `ASSUMED` alike and never built either; what would falsify it is one of them built alone, watched, and kept small.
 *   A device is matched by its vendor's name. */
static const char* const khronos_opencl2_base__zzprivate_never[][2] = {
    { "QUALCOMM", "nn__turboquant__zzabi_launch_gemv_groups__kernel" },
    { "QUALCOMM", "nn__turboquant__zzabi_launch_gemv_groups_sum__kernel" },
    { "QUALCOMM", "nn__turboquant__zzabi_launch_gemv_groups_int8__kernel" },
    { "QUALCOMM", "nn__turboquant__zzabi_launch_gemv_groups_sum_int8__kernel" },
    { "QUALCOMM", "nn__turboquant__zzabi_launch_gemv_groups_n__kernel" },
    { "QUALCOMM", "nn__expert__zzabi_launch_groups_int8__kernel" },
    { "QUALCOMM", "nn__expert__zzabi_launch_rows_sum_int8__kernel" },
    { "QUALCOMM", "khronos_opencl2_base__tile_few" },
    { "QUALCOMM", "khronos_opencl2_base__tile_wide" },
};
static inline bool khronos_opencl2_base__zzprivate_never_here(khronos_opencl2_base__state* s, uint32_t card, const char* name) {
    char vendor[128] = {0};
    if (clGetDeviceInfo(s->device[card], CL_DEVICE_VENDOR, sizeof vendor - 1u, vendor, 0) != CL_SUCCESS) return false;
    for (size_t i = 0u; i < sizeof khronos_opencl2_base__zzprivate_never / sizeof khronos_opencl2_base__zzprivate_never[0]; ++i)
        if (strstr(vendor, khronos_opencl2_base__zzprivate_never[i][0]) != 0 && strcmp(name, khronos_opencl2_base__zzprivate_never[i][1]) == 0)
            return true;
    return false;
}

/* Kernel `i` on `card`, built the first time: a refusal says why once — the device compiler's own words — and is
 * remembered, so asking again costs a lookup. Under the lock. */
static inline cl_kernel khronos_opencl2_base__zzprivate_made(khronos_opencl2_base__state* s, uint32_t card, uint32_t i) {
    if (khronos_opencl2_base__zzprivate_kernels[card] == 0) {
        khronos_opencl2_base__zzprivate_kernels[card] =
            (khronos_opencl2_base__kernel*)calloc(khronos_opencl2_base__kernel_count, sizeof(khronos_opencl2_base__kernel));
        if (khronos_opencl2_base__zzprivate_kernels[card] == 0) return 0;
    }
    khronos_opencl2_base__kernel* k = &khronos_opencl2_base__zzprivate_kernels[card][i];
    if (k->made != KHRONOS_OPENCL2_BASE__UNTRIED) return k->kernel;
    k->made = KHRONOS_OPENCL2_BASE__NONE;
    const char* name = khronos_opencl2_base__kernel_names[i];
    if (khronos_opencl2_base__zzprivate_never_here(s, card, name)) {
        fprintf(stderr, KHRONOS_OPENCL2_BASE__NAME ": %s not built on card %u, whose compiler it takes down; its door is answered "
                "without it\n", name, card);
        return 0;
    }
    const char* text[2] = { khronos_opencl2_base__kernel_bodies, khronos_opencl2_base__kernel_texts[i] };
    cl_int err = CL_SUCCESS;
    cl_program p = clCreateProgramWithSource(s->context[card], 2u, text, 0, &err);
    if (err != CL_SUCCESS) return 0;
    err = clBuildProgram(p, 1u, &s->device[card], "-cl-std=CL2.0" KHRONOS_OPENCL2_BASE__BUILD_OPTIONS, 0, 0);
    if (err != CL_SUCCESS) {
        size_t n = 0u;
        (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, 0, 0, &n);
        char* log = (char*)malloc(n + 1u);
        if (log) {
            (void)clGetProgramBuildInfo(p, s->device[card], CL_PROGRAM_BUILD_LOG, n, log, 0);
            log[n] = 0;
        }
        fprintf(stderr, KHRONOS_OPENCL2_BASE__NAME ": %s did not build on card %u (%d); nn's host body answers its door:\n%s\n",
                name, card, (int)err, log ? log : "");
        free(log);
        (void)clReleaseProgram(p);
        return 0;
    }
    /* ⛳ A KERNEL THE PROGRAM DOES NOT HAVE IS REMEMBERED TOO, as a null one — an override where the driver has no sub-groups
     *   — so asking again costs a lookup and not a build a launch */
    cl_kernel made = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS) { (void)clReleaseProgram(p); return 0; }
    k->program = p; k->kernel = made; k->made = KHRONOS_OPENCL2_BASE__BUILT;
    return made;
}

/* The kernel called `name` on `card`, made on first use; null where it did not build or there is none. */
static inline cl_kernel khronos_opencl2_base__zzprivate_kernel(khronos_opencl2_base__state* s, uint32_t card, const char* name) {
    for (uint32_t i = 0u; i < khronos_opencl2_base__kernel_count; ++i)
        if (strcmp(khronos_opencl2_base__kernel_names[i], name) == 0) return khronos_opencl2_base__zzprivate_made(s, card, i);
    return 0;
}

/* ⭐ WHETHER A CARD BUILDS ANY OF nn's DOORS — the cheapest of them, the zero fill — asked as the card is stood up
 *   (`sys.cuh`): a device whose compiler builds none of them is refused there, not taken for a card that answers
 *   nothing. */
static bool khronos_opencl2_base__zzpackage_card_builds(khronos_opencl2_base__state* s, uint32_t card) {
    return khronos_opencl2_base__zzprivate_kernel(s, card, "nn__vector__zzabi_launch_zero__kernel") != 0;
}

/* ⭐ WHETHER THE BOUND CARD HAS THE KERNEL CALLED `name` — the family's overrides ask, since theirs are left out where the
 * driver has no sub-groups (`overrides/kernels.cuh`) or its compiler refused them, and their doors are then nn's. Built
 * the first time it is asked, from the table after; under the lock. */
static inline bool khronos_opencl2_base__zzpackage_has_kernel(const char* name) {
    khronos_opencl2_base__state* s = khronos_opencl2_base__zzpackage_up();
    if (s == 0) return false;
    sys__os__lock_take(&s->lock);
    const uint32_t card = khronos_opencl2_base__bound_device();
    const bool has = card < s->count && khronos_opencl2_base__zzprivate_kernel(s, card, name) != 0;
    sys__os__lock_give(&s->lock);
    return has;
}

/* The bound card's live allocations, gathered for a launch — the ones its kernels may be handed. Under the
 * lock, into a list kept for the purpose. */
static void**  khronos_opencl2_base__zzprivate_named = 0;
static size_t  khronos_opencl2_base__zzprivate_named_room = 0u;
static inline size_t khronos_opencl2_base__zzprivate_name_card(khronos_opencl2_base__state* s, uint32_t card) {
    if (khronos_opencl2_base__zzprivate_named_room < s->live_count) {
        void** grown = (void**)realloc(khronos_opencl2_base__zzprivate_named, s->live_count * sizeof(void*));
        if (!grown) return 0u;
        khronos_opencl2_base__zzprivate_named = grown; khronos_opencl2_base__zzprivate_named_room = s->live_count;
    }
    size_t n = 0u;
    for (size_t i = 0u; i < s->live_count; ++i)
        if (s->live[i].card == card) khronos_opencl2_base__zzprivate_named[n++] = s->live[i].at;
    return n;
}

/* ⭐⭐ nn's HOST BODY BEHIND A DOOR WHOSE KERNEL DID NOT BUILD (`nn__silicon__host_body`, ▶ nn's `contracts/abi/gpu.cuh`): the
 *   card's queue drained, every allocation of the card an argument points into — a pointer argument, or one inside an
 *   argument passed by value, as a group's matrices are — mapped to the host, the row's body run there on one lane of one
 *   block (the serial loop it is on any host), and the allocations handed back. Its answer is nn's host body's, at the
 *   host's pace and the map's: a card whose memory is the host's (an integrated GPU's) maps in place, a discrete card's
 *   copies each allocation both ways. Not under the lock: the body runs, and the allocations stay, as they were. */
static inline void khronos_opencl2_base__zzprivate_on_host(khronos_opencl2_base__state* s, uint32_t card, cl_command_queue q,
                                                         const nn__silicon__host_body* host, void** args, const size_t* sizes,
                                                         uint32_t count) {
    enum { MOST = 64 };
    void* mapped[MOST]; uint32_t n = 0u; bool ok = clFinish(q) == CL_SUCCESS;
    sys__os__lock_take(&s->lock);
    for (uint32_t a = 0u; ok && a < count; ++a)
        for (size_t at = 0u; at + sizeof(uint64_t) <= sizes[a]; at += sizeof(uint64_t)) {
            uint64_t word;
            memcpy(&word, (const char*)args[a] + at, sizeof word);
            for (size_t i = 0u; i < s->live_count; ++i) {
                const khronos_opencl2_base__allocation* m = &s->live[i];
                if (m->card != card || word < (uint64_t)(uintptr_t)m->at || word >= (uint64_t)(uintptr_t)m->at + m->bytes) continue;
                uint32_t seen = 0u;
                while (seen < n && mapped[seen] != m->at) ++seen;
                if (seen == n && n < MOST) {
                    ok = clEnqueueSVMMap(q, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, m->at, m->bytes, 0u, 0, 0) == CL_SUCCESS;
                    if (ok) mapped[n++] = m->at;
                }
                break;
            }
        }
    sys__os__lock_give(&s->lock);
    if (ok) host->body(host->args);
    for (uint32_t i = 0u; i < n; ++i) ok = clEnqueueSVMUnmap(q, mapped[i], 0u, 0, 0) == CL_SUCCESS && ok;
    ok = clFinish(q) == CL_SUCCESS && ok;
    if (!ok) fprintf(stderr, KHRONOS_OPENCL2_BASE__NAME ": nn's host body could not reach card %u's memory\n", card);
}

/* Start the kernel called `name` on `blocks` blocks of `threads` threads. `kernel` is nn's host body behind the door
 * (`nn__silicon__host_body`) — this family compiles no kernel of its own to take an address of — or null, an override's
 * launch, which asks first whether its kernel was made (▶ `zzpackage_has_kernel`).
 * ⛳ UNDER THE LOCK, because a kernel's arguments belong to the kernel object and not to the launch: two
 *   threads setting them at once would launch each other's.
 * ⛔ AND THE CARD'S QUEUE DRAINED WHEN ITS LIVE LIST HAS CHANGED SINCE ITS LAST LAUNCH: a kernel object launched again
 *   with another list of allocations, while its last launch still waits in the queue, made that launch run wrong —
 *   `MEASURED` on node03's MI50, ROCm 6.1.3's OpenCL: GLM's router copied a chunk's rows into the hands, the tier's note
 *   made its landing, and its gather, the same kernel, ran before the copy had; in 1 to 4 runs of 8 the copy's rows were
 *   lost. A kernel object made afresh each launch, or a drain anywhere between the two, and 8 of 8 were right
 *   (measurements/2026-10-26_nn126_opencl2_node03.md). A list changes when memory is allocated or freed, a few times a
 *   layer at most, and the drain waits only for what the next read would.
 * ⭐ A KERNEL THAT DID NOT BUILD IS NEVER LAUNCHED: its door is nn's host body (▶ above). */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    khronos_opencl2_base__state* s = khronos_opencl2_base__zzpackage_up();
    cl_command_queue q = khronos_opencl2_base__zzpackage_queue();
    if (s == 0 || q == 0) return;
    sys__os__lock_take(&s->lock);
    const uint32_t card = khronos_opencl2_base__bound_device();
    cl_kernel k = card < s->count ? khronos_opencl2_base__zzprivate_kernel(s, card, name) : 0;
    if (k == 0 && kernel != 0 && card < s->count) {
        sys__os__lock_give(&s->lock);
        khronos_opencl2_base__zzprivate_on_host(s, card, q, (const nn__silicon__host_body*)kernel, args, sizes, count);
        return;
    }
    bool ok = k != 0;
    if (ok && s->launched[card] != s->changed[card]) {
        ok = clFinish(q) == CL_SUCCESS;
        s->launched[card] = s->changed[card];
    }
    for (uint32_t i = 0u; ok && i < count; ++i) ok = clSetKernelArg(k, i, sizes[i], args[i]) == CL_SUCCESS;
    const size_t named = ok ? khronos_opencl2_base__zzprivate_name_card(s, card) : 0u;
    if (ok && named)
        ok = clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_SVM_PTRS, named * sizeof(void*), khronos_opencl2_base__zzprivate_named) == CL_SUCCESS;
    if (ok) {
        const size_t global = (size_t)blocks * (size_t)threads, local = (size_t)threads;
        ok = clEnqueueNDRangeKernel(q, k, 1u, 0, &global, &local, 0u, 0, 0) == CL_SUCCESS;
    }
    sys__os__lock_give(&s->lock);
    if (!ok) fprintf(stderr, KHRONOS_OPENCL2_BASE__NAME ": %s did not launch\n", name);
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

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_BASE_NN_CUH */
