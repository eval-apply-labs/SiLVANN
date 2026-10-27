#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_SYS_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_SYS_CUH
/* ══ sys's DOORS — khronos_opencl1_igpu ════════════════════════════════════════════════════════════════════
 *
 * sys's doors through OpenCL 1.2, which has no shared virtual memory: a kernel reaches memory only through
 * the buffers it is handed, and a buffer has no address the host can do arithmetic on. Everything `sys` does
 * with a card address — offsets, the heap's arithmetic, a pointer in a kernel's arguments or in a table of
 * addresses — wants one. So this family makes its own:
 *
 * ⭐ AN ADDRESS IS AN ARENA AND AN OFFSET. Memory comes from a few large buffers per card, the ARENAS, each
 *   handing out spans of itself; the address of a span is a number with the card and the arena in its high
 *   bits and the offset in its low 40:
 *
 *       bit 62 set · card (bits 48-55) · arena (bits 40-47) · offset (bits 0-39)
 *
 *   An offset within a span is an offset within the address, so the arithmetic is unchanged. Bit 62 is never
 *   set in a pointer of the program's own, which is how a copy tells the card's end from the host's.
 * ⭐ EVERY KERNEL IS HANDED ITS CARD'S ARENAS (`nn.cuh`), and the program the generator writes turns each
 *   address into the arena's pointer plus the offset (`scripts/opencl1_kernel_source.py`) — the two sides
 *   of one encoding, so `KHRONOS_OPENCL1_IGPU__ARENAS` and the generator's `ARENAS` are one number.
 * ⛳ AN ARENA IS 1 GiB — the card's largest allocation if that is less — or one span alone when a span is larger.
 *   ⛳ A SPAN LARGER THAN THE CARD SAYS IT TAKES IS ASKED FOR ANYWAY, and kept only if filling it on the card works:
 *   the engine's heap is one span holding the weights (14.9 GB for the 27B), and `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is a
 *   floor the runtime promises, not always its ceiling. `MEASURED` on the RTX 5060 Ti (NVIDIA's OpenCL, which says
 *   3.87 GiB): an 8 GiB buffer made, filled, and written by a kernel 3 bytes from its end. A runtime that keeps to
 *   its number refuses the span, and the boot says so. An arena whose spans are all given back is given back.
 * ⭐ MAPPED HOST RAM IS TWO COPIES KEPT IN STEP, because a 1.2 card has no memory both sides see: the program's
 *   copy in its own RAM and the card's in a span of an arena. `MEASURED` on the RTX 5060 Ti (NVIDIA's OpenCL,
 *   precision, `measurements/2026-10-13_khronos_opencl1.md`): a kernel's store into a buffer made with
 *   `CL_MEM_ALLOC_HOST_PTR` and mapped, or with `CL_MEM_USE_HOST_PTR`, is not in the host's view after the queue
 *   drains — 0 of 4,096 words — and the card says so (`CL_DEVICE_HOST_UNIFIED_MEMORY` 0). So:
 *     · a launch handed an address inside one (an argument, or a word of a table argument) first writes the
 *       program's copy to the card, unless the card's copy is the newer; the card's copy is then the newer;
 *     · waiting for the card (`compute_completed`) reads every newer card copy back into the program's;
 *     · a copy or a fill with an end inside one works on the program's copy, after reading the card's back.
 *   ⛳ nn's read of a value verb's answer (`nn__buffer__read`) waits for the card when its watch of the program's
 *   copy runs out, which is what brings the answer across.
 *   ⛔ THE STEP IS THE WHOLE ALLOCATION, and that is what this family's rate pays for: nn's RAM span (the 27B's
 *   embedding table, 1.27 GB) crosses whole both ways once a token, its one reader a kernel decoding a row of it, which
 *   a launch cannot tell from a writer — `MEASURED` 1.61 tokens/s against CUDA's 21 on the same bodies, same record. A
 *   model with the experts on the CPU decodes the row there instead (Qwen's `embed_on`), and the span holds nothing a
 *   token reads.
 * ⭐⭐ ON A CARD WHOSE MEMORY IS THE HOST'S, ONE COPY (NN-73) — an integrated GPU's: a laptop's, an APU's. ⚖
 *   2026-10-20 *"so also pcs with an igpu get the benefit"*. Where the card says `CL_DEVICE_HOST_UNIFIED_MEMORY`, a span
 *   of mapped RAM is an arena of its own, a buffer made over the program's copy (`CL_MEM_USE_HOST_PTR`), and the steps
 *   across are OpenCL's own hand-over of it — unmapped before a kernel is handed it, mapped again when the card is
 *   waited for — which move nothing where the memory is one. Kept only where mapping it answers the program's copy's
 *   own address (the runtime did not make a copy of its own); otherwise, or with no slot free among the 32, the span
 *   is the two copies above. `SILVANN_OPENCL1_UNIFIED=0` keeps every span two copies, to measure one against the other.
 *
 * ⛔⛔ ONE CONTEXT PER CARD, AND AN ARENA BELONGS TO THE CARD IT WAS MADE ON — as `khronos_opencl2`'s
 *   allocations do, and for its reason: a context over several cards puts a buffer in host RAM.
 * ⛳ And one in-order queue per card, so work on a card runs in the order it was asked for, as a HIP
 *   stream does. A thread's bound card is its own (`__thread`), and device 0 until a thread binds another. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#define CL_TARGET_OPENCL_VERSION 120
#include "../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <CL/cl.h>
#include "../../operating_systems/os.cuh"   /* threads, locks and conditions: the system's */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "includes.cuh"   /* which cards are this family's */

#define KHRONOS_OPENCL1_IGPU__DEVICES_MAX   16u
#define KHRONOS_OPENCL1_IGPU__ARENAS        32u               /* a kernel's arena arguments: the generator's ARENAS */
#define KHRONOS_OPENCL1_IGPU__ARENA_BYTES   (1ull << 30)
#define KHRONOS_OPENCL1_IGPU__ALIGN         256ull            /* every span starts on this */
#define KHRONOS_OPENCL1_IGPU__TAG           (1ull << 62)
#define KHRONOS_OPENCL1_IGPU__OFFSET_MASK   ((1ull << 40) - 1ull)
/* `cl_nv_device_attribute_query`'s two names, as NVIDIA's `cl_ext.h` spells them */
#define KHRONOS_OPENCL1_IGPU__NV_MAJOR      0x4000
#define KHRONOS_OPENCL1_IGPU__NV_MINOR      0x4001

typedef struct khronos_opencl1_igpu__span {
    uint64_t at, bytes;
} khronos_opencl1_igpu__span;

typedef struct khronos_opencl1_igpu__arena {
    cl_mem                 buffer;                          /* null: the slot is empty */
    uint64_t               bytes;
    uint64_t               used;                            /* bytes handed out */
    khronos_opencl1_igpu__span* free;                            /* its free spans, by offset */
    size_t                 free_count, free_room;
    bool                   host;                            /* one span of mapped RAM, over the program's copy (NN-73) */
    bool                   host_holds;                      /* ...mapped to the host now: no kernel may be handed it */
} khronos_opencl1_igpu__arena;

/* A span handed out: its address and its size, and for mapped RAM the program's copy and which copy is the newer. */
typedef struct khronos_opencl1_igpu__allocation {
    uint64_t at;
    uint64_t bytes;
    uint8_t* host;              /* mapped RAM: the program's copy; null for card memory */
    bool     card_newer;        /* mapped RAM: a kernel was handed it since the copies were last in step */
} khronos_opencl1_igpu__allocation;

typedef struct khronos_opencl1_igpu__state {
    cl_uint                 count;                                       /* cards of this family, in platform order */
    cl_device_id            device[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];
    cl_context              context[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];       /* one each: ▶ the note above */
    cl_command_queue        queue[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];         /* one in-order queue each */
    uint64_t                largest[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];       /* the card's largest one buffer */
    bool                    unified[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];       /* its memory the host's (NN-73) */
    khronos_opencl1_igpu__arena  arena[KHRONOS_OPENCL1_IGPU__DEVICES_MAX][KHRONOS_OPENCL1_IGPU__ARENAS];
    uint64_t                generation[KHRONOS_OPENCL1_IGPU__DEVICES_MAX];    /* moves when a card's arenas change */
    khronos_opencl1_igpu__allocation* live;                                   /* every span of card memory not yet given back */
    size_t                  live_count, live_room;
    khronos_opencl1_igpu__allocation* mapped;                                 /* every span of mapped RAM, the same */
    size_t                  mapped_count, mapped_room;
} khronos_opencl1_igpu__state;

static khronos_opencl1_igpu__state khronos_opencl1_igpu__zzprivate_state;
static sys__os__lock          khronos_opencl1_igpu__zzpackage_lock = SYS__OS__LOCK_INIT;   /* the arenas, the live list, a kernel's arguments */
static sys__os__once          khronos_opencl1_igpu__zzprivate_once = SYS__OS__ONCE_INIT;
static __thread uint32_t      khronos_opencl1_igpu__zzprivate_bound = 0u;

/* A device's name as the roster's rows spell it: an NVIDIA card's compute capability, any other's device name. */
static inline bool khronos_opencl1_igpu__zzprivate_name(cl_device_id d, char* name, size_t room) {
    cl_uint major = 0u, minor = 0u;
    if (clGetDeviceInfo(d, KHRONOS_OPENCL1_IGPU__NV_MAJOR, sizeof major, &major, 0) == CL_SUCCESS &&
        clGetDeviceInfo(d, KHRONOS_OPENCL1_IGPU__NV_MINOR, sizeof minor, &minor, 0) == CL_SUCCESS) {
        snprintf(name, room, "sm_%u%u", major, minor);
        return true;
    }
    memset(name, 0, room);
    return clGetDeviceInfo(d, CL_DEVICE_NAME, room - 1u, name, 0) == CL_SUCCESS;
}

/* Whether a device compiles OpenCL C 1.2 — what the generated program is written in. */
static inline bool khronos_opencl1_igpu__zzprivate_suits(cl_device_id d) {
    char v[128] = {0};
    unsigned major = 0u, minor = 0u;
    if (clGetDeviceInfo(d, CL_DEVICE_OPENCL_C_VERSION, sizeof v - 1u, v, 0) != CL_SUCCESS) return false;
    if (sscanf(v, "OpenCL C %u.%u", &major, &minor) != 2) return false;
    return major > 1u || (major == 1u && minor >= 2u);
}

/* Find this family's cards on the first platform that has any, and stand up a context and a queue for
 * each. Once per process; a machine with none leaves the count at zero, and a card whose context or queue
 * will not stand up takes the family's cards with it — a partial set would renumber the rest. */
/* ⛳ A DEVELOPER'S DEVICE: `SILVANN_OPENCL1_TAKE` names one by the start of its name, of any type, beside the cards the
 *   family includes — a CPU's OpenCL (PoCL) is the nearest to an integrated GPU's on a box with none: its memory the
 *   host's (`CL_DEVICE_HOST_UNIFIED_MEMORY`), its OpenCL C 1.2. Never in a release's use; the checks' alone. */
static void khronos_opencl1_igpu__zzprivate_stand_up(void) {
    khronos_opencl1_igpu__state* s = &khronos_opencl1_igpu__zzprivate_state;
    const char* take = getenv("SILVANN_OPENCL1_TAKE");
    cl_platform_id platforms[8]; cl_uint np = 0u;
    if (clGetPlatformIDs(8u, platforms, &np) != CL_SUCCESS) return;
    for (cl_uint p = 0u; p < np && p < 8u && s->count == 0u; ++p) {
        cl_device_id all[KHRONOS_OPENCL1_IGPU__DEVICES_MAX]; cl_uint nd = 0u;
        const cl_device_type type = take && take[0] ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU;
        if (clGetDeviceIDs(platforms[p], type, KHRONOS_OPENCL1_IGPU__DEVICES_MAX, all, &nd) != CL_SUCCESS) continue;
        for (cl_uint d = 0u; d < nd && d < KHRONOS_OPENCL1_IGPU__DEVICES_MAX; ++d) {
            char name[256];
            if (!khronos_opencl1_igpu__zzprivate_name(all[d], name, sizeof name)) continue;
            const bool taken = take && take[0] ? strncmp(name, take, strlen(take)) == 0 : khronos_opencl1_igpu__includes(name);
            if (taken && khronos_opencl1_igpu__zzprivate_suits(all[d])) s->device[s->count++] = all[d];
        }
    }
    if (s->count == 0u) return;
    cl_int err = CL_SUCCESS;
    for (cl_uint d = 0u; d < s->count; ++d) {
        cl_ulong largest = 0u;
        if (clGetDeviceInfo(s->device[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof largest, &largest, 0) != CL_SUCCESS) { s->count = 0u; return; }
        s->largest[d] = (uint64_t)largest;
        cl_bool unified = CL_FALSE;
        const char* say = getenv("SILVANN_OPENCL1_UNIFIED");
        s->unified[d] = clGetDeviceInfo(s->device[d], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, 0) == CL_SUCCESS
                     && unified == CL_TRUE && !(say && say[0] == '0');
        s->context[d] = clCreateContext(0, 1u, &s->device[d], 0, 0, &err);
        if (err != CL_SUCCESS) { s->count = 0u; return; }
        cl_command_queue_properties props = 0;
        s->queue[d] = clCreateCommandQueue(s->context[d], s->device[d], props, &err);
        if (err != CL_SUCCESS) { s->count = 0u; return; }
    }
}

static inline khronos_opencl1_igpu__state* khronos_opencl1_igpu__zzpackage_up(void) {
    sys__os__once_run(&khronos_opencl1_igpu__zzprivate_once, khronos_opencl1_igpu__zzprivate_stand_up);
    return khronos_opencl1_igpu__zzprivate_state.count > 0u ? &khronos_opencl1_igpu__zzprivate_state : 0;
}

/* The queue of the card the calling thread is on. */
static inline cl_command_queue khronos_opencl1_igpu__zzpackage_queue(void) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    return (s && khronos_opencl1_igpu__zzprivate_bound < s->count) ? s->queue[khronos_opencl1_igpu__zzprivate_bound] : 0;
}

/* ── AN ADDRESS ──────────────────────────────────────────────────────────────────────────────────────── */
static inline uint64_t khronos_opencl1_igpu__zzprivate_address(uint32_t card, uint32_t slot, uint64_t offset) {
    return KHRONOS_OPENCL1_IGPU__TAG | ((uint64_t)card << 48) | ((uint64_t)slot << 40) | offset;
}

/* Where `at` is, when it is an address this family handed out: its card, its arena and the offset in it. */
static inline bool khronos_opencl1_igpu__zzprivate_where(const void* at, uint32_t* card, uint32_t* slot, uint64_t* offset) {
    const uint64_t a = (uint64_t)(uintptr_t)at;
    if ((a >> 62) != 1ull) return false;
    *card = (uint32_t)((a >> 48) & 0xFFull); *slot = (uint32_t)((a >> 40) & 0xFFull); *offset = a & KHRONOS_OPENCL1_IGPU__OFFSET_MASK;
    return *card < KHRONOS_OPENCL1_IGPU__DEVICES_MAX && *slot < KHRONOS_OPENCL1_IGPU__ARENAS;
}

/* ── AN ARENA'S SPANS ────────────────────────────────────────────────────────────────────────────────── */
static inline bool khronos_opencl1_igpu__zzprivate_take(khronos_opencl1_igpu__arena* a, uint64_t bytes, uint64_t* at) {
    for (size_t i = 0u; i < a->free_count; ++i) {
        khronos_opencl1_igpu__span* f = &a->free[i];
        if (f->bytes < bytes) continue;
        *at = f->at;
        f->at += bytes; f->bytes -= bytes;
        if (f->bytes == 0u) { memmove(f, f + 1, (a->free_count - i - 1u) * sizeof *f); --a->free_count; }
        a->used += bytes;
        return true;
    }
    return false;
}

/* A span back into its arena's free list, joined to the free spans either side of it. */
static inline bool khronos_opencl1_igpu__zzprivate_give(khronos_opencl1_igpu__arena* a, uint64_t at, uint64_t bytes) {
    size_t i = 0u;
    while (i < a->free_count && a->free[i].at < at) ++i;
    const bool joins_before = i > 0u && a->free[i - 1u].at + a->free[i - 1u].bytes == at;
    const bool joins_after  = i < a->free_count && at + bytes == a->free[i].at;
    if (joins_before && joins_after) {
        a->free[i - 1u].bytes += bytes + a->free[i].bytes;
        memmove(&a->free[i], &a->free[i + 1u], (a->free_count - i - 1u) * sizeof *a->free); --a->free_count;
    } else if (joins_before) {
        a->free[i - 1u].bytes += bytes;
    } else if (joins_after) {
        a->free[i].at = at; a->free[i].bytes += bytes;
    } else {
        if (a->free_count == a->free_room) {
            size_t room = a->free_room ? 2u * a->free_room : 16u;
            khronos_opencl1_igpu__span* grown = (khronos_opencl1_igpu__span*)realloc(a->free, room * sizeof *grown);
            if (!grown) return false;
            a->free = grown; a->free_room = room;
        }
        memmove(&a->free[i + 1u], &a->free[i], (a->free_count - i) * sizeof *a->free);
        a->free[i].at = at; a->free[i].bytes = bytes; ++a->free_count;
    }
    a->used -= bytes;
    return true;
}

/* Zero `bytes` of `b` from `offset`, with the widest pattern both divide — a byte at a time is the slow fill. */
static inline bool khronos_opencl1_igpu__zzprivate_fill(cl_command_queue q, cl_mem b, size_t offset, size_t bytes) {
    static const cl_uchar zero[16] = {0};
    size_t width = 16u;
    while (width > 1u && (offset % width != 0u || bytes % width != 0u)) width /= 2u;
    return clEnqueueFillBuffer(q, b, zero, width, offset, bytes, 0u, 0, 0) == CL_SUCCESS && clFinish(q) == CL_SUCCESS;
}

/* A new arena in slot `slot` of `card`, of `bytes`.
 * ⛳ IT IS FILLED ONCE HERE, so the runtime places it on the card now: an allocation it only places at first use
 *   would fail at a kernel's launch instead of here, where the caller can be told. */
static inline bool khronos_opencl1_igpu__zzprivate_arena(khronos_opencl1_igpu__state* s, uint32_t card, uint32_t slot, uint64_t bytes) {
    khronos_opencl1_igpu__arena* a = &s->arena[card][slot];
    cl_int err = CL_SUCCESS;
    cl_mem b = clCreateBuffer(s->context[card], CL_MEM_READ_WRITE, (size_t)bytes, 0, &err);
    if (err != CL_SUCCESS || b == 0) return false;
    if (!khronos_opencl1_igpu__zzprivate_fill(s->queue[card], b, 0u, (size_t)bytes)) { (void)clReleaseMemObject(b); return false; }
    a->free = (khronos_opencl1_igpu__span*)malloc(16u * sizeof(khronos_opencl1_igpu__span));
    if (a->free == 0) { (void)clReleaseMemObject(b); return false; }
    a->free_room = 16u; a->free_count = 1u; a->free[0].at = 0u; a->free[0].bytes = bytes;
    a->buffer = b; a->bytes = bytes; a->used = 0u;
    ++s->generation[card];
    return true;
}

/* ⛳ ITS CARD'S QUEUE IS DRAINED FIRST: a kernel still queued may be using the arena. */
static inline void khronos_opencl1_igpu__zzprivate_drop_arena(khronos_opencl1_igpu__state* s, uint32_t card, uint32_t slot) {
    khronos_opencl1_igpu__arena* a = &s->arena[card][slot];
    (void)clFinish(s->queue[card]);
    if (a->host && a->host_holds) {                         /* a buffer is unmapped before it is released */
        void* host = 0;
        (void)clGetMemObjectInfo(a->buffer, CL_MEM_HOST_PTR, sizeof host, &host, 0);
        (void)clEnqueueUnmapMemObject(s->queue[card], a->buffer, host, 0u, 0, 0);
        (void)clFinish(s->queue[card]);
    }
    (void)clReleaseMemObject(a->buffer);
    free(a->free);
    memset(a, 0, sizeof *a);
    ++s->generation[card];
}

/* Room for one more entry in a list of allocations. */
static inline bool khronos_opencl1_igpu__zzprivate_room(khronos_opencl1_igpu__allocation** list, size_t count, size_t* room) {
    if (count < *room) return true;
    size_t more = *room ? 2u * *room : 64u;
    khronos_opencl1_igpu__allocation* grown = (khronos_opencl1_igpu__allocation*)realloc(*list, more * sizeof *grown);
    if (!grown) return false;
    *list = grown; *room = more;
    return true;
}

/* A span of `bytes` on the bound card: the first arena with room, else a new one. Under the lock. */
static inline uint64_t khronos_opencl1_igpu__zzprivate_span(khronos_opencl1_igpu__state* s, uint32_t card, uint64_t want) {
    uint64_t offset = 0u;
    for (uint32_t k = 0u; k < KHRONOS_OPENCL1_IGPU__ARENAS; ++k) {
        khronos_opencl1_igpu__arena* a = &s->arena[card][k];
        if (a->buffer && khronos_opencl1_igpu__zzprivate_take(a, want, &offset)) return khronos_opencl1_igpu__zzprivate_address(card, k, offset);
    }
    for (uint32_t k = 0u; k < KHRONOS_OPENCL1_IGPU__ARENAS; ++k) {
        if (s->arena[card][k].buffer) continue;
        const uint64_t usual = KHRONOS_OPENCL1_IGPU__ARENA_BYTES < s->largest[card] ? KHRONOS_OPENCL1_IGPU__ARENA_BYTES : s->largest[card];
        const uint64_t size = want > usual ? want : usual;
        /* a card with less than an arena's room left still takes a span that fits what it has */
        const bool made = khronos_opencl1_igpu__zzprivate_arena(s, card, k, size) ||
                          (size > want && khronos_opencl1_igpu__zzprivate_arena(s, card, k, want));
        if (made && khronos_opencl1_igpu__zzprivate_take(&s->arena[card][k], want, &offset))
            return khronos_opencl1_igpu__zzprivate_address(card, k, offset);
        return 0u;
    }
    return 0u;
}

/* A span given back to its arena; an arena left empty goes back to the card. Under the lock. */
static inline void khronos_opencl1_igpu__zzprivate_unspan(khronos_opencl1_igpu__state* s, uint64_t at, uint64_t bytes) {
    uint32_t card = 0u, slot = 0u; uint64_t offset = 0u;
    if (!khronos_opencl1_igpu__zzprivate_where((const void*)(uintptr_t)at, &card, &slot, &offset) || card >= s->count) return;
    khronos_opencl1_igpu__arena* a = &s->arena[card][slot];
    if (a->buffer && khronos_opencl1_igpu__zzprivate_give(a, offset, bytes) && a->used == 0u)
        khronos_opencl1_igpu__zzprivate_drop_arena(s, card, slot);
}

/* ⭐ MAPPED RAM AS AN ARENA OF ITS OWN, over the program's copy (NN-73): a buffer made on `host` itself, and mapped to
 *   the host at once — the program's copy is the one it writes first. The address of slot `slot`'s first byte, or 0
 *   when the card would copy it anyway (mapping it answered another address) and the span is two copies instead. Under
 *   the lock. */
static inline uint64_t khronos_opencl1_igpu__zzprivate_host_arena(khronos_opencl1_igpu__state* s, uint32_t card, uint8_t* host,
                                                                  uint64_t bytes) {
    for (uint32_t k = 0u; k < KHRONOS_OPENCL1_IGPU__ARENAS; ++k) {
        khronos_opencl1_igpu__arena* a = &s->arena[card][k];
        if (a->buffer) continue;
        cl_int err = CL_SUCCESS;
        cl_mem b = clCreateBuffer(s->context[card], CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (size_t)bytes, host, &err);
        if (err != CL_SUCCESS || b == 0) return 0u;
        void* seen = clEnqueueMapBuffer(s->queue[card], b, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0u, (size_t)bytes, 0u, 0, 0, &err);
        if (err != CL_SUCCESS || seen != (void*)host) {
            if (err == CL_SUCCESS) { (void)clEnqueueUnmapMemObject(s->queue[card], b, seen, 0u, 0, 0); (void)clFinish(s->queue[card]); }
            (void)clReleaseMemObject(b);
            return 0u;
        }
        a->free = (khronos_opencl1_igpu__span*)malloc(16u * sizeof(khronos_opencl1_igpu__span));
        if (a->free == 0) {
            (void)clEnqueueUnmapMemObject(s->queue[card], b, seen, 0u, 0, 0); (void)clFinish(s->queue[card]);
            (void)clReleaseMemObject(b);
            return 0u;
        }
        a->free_room = 16u; a->free_count = 0u;            /* the one span is the whole of it, handed out */
        a->buffer = b; a->bytes = bytes; a->used = bytes; a->host = true; a->host_holds = true;
        ++s->generation[card];
        return khronos_opencl1_igpu__zzprivate_address(card, k, 0u);
    }
    return 0u;
}

/* A span of `bytes` on the bound card, and for mapped RAM the program's copy beside it, in `*host`. */
static inline void* khronos_opencl1_igpu__zzprivate_allocate(size_t bytes, bool mapped, void** host) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    const uint32_t card = khronos_opencl1_igpu__zzprivate_bound;
    if (s == 0 || card >= s->count) return 0;
    const uint64_t want = (((uint64_t)(bytes ? bytes : 1u)) + KHRONOS_OPENCL1_IGPU__ALIGN - 1ull) & ~(KHRONOS_OPENCL1_IGPU__ALIGN - 1ull);
    if (want > KHRONOS_OPENCL1_IGPU__OFFSET_MASK) return 0;
    uint8_t* copy = 0;
    const uint64_t pages = (want + 4095ull) & ~4095ull;
    if (mapped) {
        copy = (uint8_t*)sys__os__memory_aligned(4096u, (size_t)pages);
        if (copy == 0) return 0;
        memset(copy, 0, (size_t)pages);
    }
    uint64_t at = 0u;
    sys__os__lock_take(&khronos_opencl1_igpu__zzpackage_lock);
    khronos_opencl1_igpu__allocation** list = mapped ? &s->mapped : &s->live;
    size_t* count = mapped ? &s->mapped_count : &s->live_count;
    size_t* room  = mapped ? &s->mapped_room  : &s->live_room;
    if (khronos_opencl1_igpu__zzprivate_room(list, *count, room)) {
        if (mapped && s->unified[card]) at = khronos_opencl1_igpu__zzprivate_host_arena(s, card, copy, pages);
        if (at == 0u) at = khronos_opencl1_igpu__zzprivate_span(s, card, want);
    }
    if (at) {
        khronos_opencl1_igpu__allocation* l = &(*list)[(*count)++];
        l->at = at; l->bytes = want; l->host = copy; l->card_newer = false;
    }
    sys__os__lock_give(&khronos_opencl1_igpu__zzpackage_lock);
    if (at == 0u) { sys__os__memory_aligned_free(copy); return 0; }
    if (host) *host = copy;
    return (void*)(uintptr_t)at;
}

/* A span given back, named by its address — or, for mapped RAM, by the program's copy. */
static inline void khronos_opencl1_igpu__zzprivate_release(const void* at, bool mapped) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    if (s == 0 || at == 0) return;
    sys__os__lock_take(&khronos_opencl1_igpu__zzpackage_lock);
    khronos_opencl1_igpu__allocation* list = mapped ? s->mapped : s->live;
    size_t* count = mapped ? &s->mapped_count : &s->live_count;
    for (size_t i = 0u; i < *count; ++i) {
        const khronos_opencl1_igpu__allocation l = list[i];
        if (mapped ? (const void*)l.host != at : (const void*)(uintptr_t)l.at != at) continue;
        list[i] = list[--*count];
        uint32_t card = 0u, slot = 0u; uint64_t offset = 0u;
        if (mapped && khronos_opencl1_igpu__zzprivate_where((const void*)(uintptr_t)l.at, &card, &slot, &offset) && card < s->count
         && s->arena[card][slot].host)
            khronos_opencl1_igpu__zzprivate_drop_arena(s, card, slot);           /* its own arena, whole (NN-73) */
        else
            khronos_opencl1_igpu__zzprivate_unspan(s, l.at, l.bytes);
        sys__os__memory_aligned_free(l.host);
        break;
    }
    sys__os__lock_give(&khronos_opencl1_igpu__zzpackage_lock);
}

/* ── MAPPED RAM'S TWO COPIES ─────────────────────────────────────────────────────────────────────────── */
/* The mapped allocation `at` falls inside, or null. Under the lock. */
static inline khronos_opencl1_igpu__allocation* khronos_opencl1_igpu__zzprivate_mapped(khronos_opencl1_igpu__state* s, uint64_t at) {
    if ((at >> 62) != 1ull) return 0;
    for (size_t i = 0u; i < s->mapped_count; ++i)
        if (at >= s->mapped[i].at && at < s->mapped[i].at + s->mapped[i].bytes) return &s->mapped[i];
    return 0;
}

/* One way across: the program's copy to the card's (`up`), or back. Blocking, in `q`. Under the lock. */
static inline bool khronos_opencl1_igpu__zzprivate_across(khronos_opencl1_igpu__state* s, khronos_opencl1_igpu__allocation* m,
                                                     cl_command_queue q, bool up) {
    uint32_t card = 0u, slot = 0u; uint64_t offset = 0u;
    if (!khronos_opencl1_igpu__zzprivate_where((const void*)(uintptr_t)m->at, &card, &slot, &offset) || card >= s->count) return false;
    khronos_opencl1_igpu__arena* a = &s->arena[card][slot];
    const cl_mem b = a->buffer;
    if (a->host) {                                          /* NN-73: handed over, not copied */
        if (up) {
            if (a->host_holds && clEnqueueUnmapMemObject(q, b, m->host, 0u, 0, 0) != CL_SUCCESS) return false;
            a->host_holds = false;
            return true;
        }
        cl_int err = CL_SUCCESS;
        void* seen = a->host_holds ? (void*)m->host
                   : clEnqueueMapBuffer(q, b, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0u, (size_t)a->bytes, 0u, 0, 0, &err);
        if (err != CL_SUCCESS || seen != (void*)m->host) return false;
        a->host_holds = true;
        m->card_newer = false;
        return true;
    }
    const bool ok = up ? clEnqueueWriteBuffer(q, b, CL_TRUE, (size_t)offset, (size_t)m->bytes, m->host, 0u, 0, 0) == CL_SUCCESS
                       : clEnqueueReadBuffer(q, b, CL_TRUE, (size_t)offset, (size_t)m->bytes, m->host, 0u, 0, 0) == CL_SUCCESS;
    if (ok && !up) m->card_newer = false;
    return ok;
}

/* ⭐ BEFORE A LAUNCH: every word of every argument that is an address inside mapped RAM names it to the kernel. Its
 *   program's copy goes to the card first, unless the card's is the newer, and the card's is the newer after. Under
 *   the lock, in the launch's queue — so the write lands after the work before it and before the kernel. */
static inline bool khronos_opencl1_igpu__zzpackage_launching(khronos_opencl1_igpu__state* s, cl_command_queue q,
                                                        void** args, const size_t* sizes, uint32_t count) {
    if (s->mapped_count == 0u) return true;
    for (uint32_t i = 0u; i < count; ++i)
        for (size_t w = 0u; w + sizeof(uint64_t) <= sizes[i]; w += sizeof(uint64_t)) {
            uint64_t word = 0u;
            memcpy(&word, (const uint8_t*)args[i] + w, sizeof word);
            khronos_opencl1_igpu__allocation* m = khronos_opencl1_igpu__zzprivate_mapped(s, word);
            if (m == 0) continue;
            if (!m->card_newer && !khronos_opencl1_igpu__zzprivate_across(s, m, q, true)) return false;
            m->card_newer = true;
        }
    return true;
}

/* Every mapped allocation of `card` whose card copy is the newer, read back — after the card's queue has drained. */
static inline bool khronos_opencl1_igpu__zzprivate_catch_up(khronos_opencl1_igpu__state* s, uint32_t card) {
    bool ok = true;
    sys__os__lock_take(&khronos_opencl1_igpu__zzpackage_lock);
    for (size_t i = 0u; i < s->mapped_count; ++i) {
        khronos_opencl1_igpu__allocation* m = &s->mapped[i];
        if (m->card_newer && (uint32_t)((m->at >> 48) & 0xFFull) == card)
            ok = khronos_opencl1_igpu__zzprivate_across(s, m, s->queue[card], false) && ok;
    }
    sys__os__lock_give(&khronos_opencl1_igpu__zzpackage_lock);
    return ok;
}

/* ── ONE END OF A COPY ───────────────────────────────────────────────────────────────────────────────── */
typedef struct khronos_opencl1_igpu__end {
    bool     card;          /* a span of card memory: `buffer` at `offset` on `owner` */
    uint32_t owner;
    cl_mem   buffer;
    size_t   offset;
    uint8_t* host;          /* otherwise the program's memory — mapped RAM's copy, or its own */
} khronos_opencl1_igpu__end;

/* What `at` is: card memory, or memory the program reaches — mapped RAM's copy, brought up to date first. */
static inline bool khronos_opencl1_igpu__zzprivate_end(khronos_opencl1_igpu__state* s, const void* at, khronos_opencl1_igpu__end* e) {
    uint32_t card = 0u, slot = 0u; uint64_t offset = 0u;
    memset(e, 0, sizeof *e);
    e->owner = 0xFFFFFFFFu;
    if (!khronos_opencl1_igpu__zzprivate_where(at, &card, &slot, &offset)) { e->host = (uint8_t*)(uintptr_t)at; return true; }
    if (card >= s->count) return false;
    bool ok = true;
    sys__os__lock_take(&khronos_opencl1_igpu__zzpackage_lock);
    const khronos_opencl1_igpu__arena a = s->arena[card][slot];
    khronos_opencl1_igpu__allocation* m = khronos_opencl1_igpu__zzprivate_mapped(s, (uint64_t)(uintptr_t)at);
    if (m) {
        if (m->card_newer) ok = clFinish(s->queue[card]) == CL_SUCCESS && khronos_opencl1_igpu__zzprivate_across(s, m, s->queue[card], false);
        e->host = m->host + ((uint64_t)(uintptr_t)at - m->at);
    }
    sys__os__lock_give(&khronos_opencl1_igpu__zzpackage_lock);
    e->owner = card;
    if (m) return ok;
    if (a.buffer == 0 || offset > a.bytes) return false;
    e->card = true; e->buffer = a.buffer; e->offset = (size_t)offset;
    return true;
}

/* ⛳ BLOCKING, AND IN THE QUEUE OF THE CARD THAT OWNS AN END — the card end first — so a copy waits for the
 *   work asked for before it on that card, as `hipMemcpy` does; `channel` instead for the side channel's,
 *   which does not wait. Mapped RAM is copied through the program's copy. */
static inline bool khronos_opencl1_igpu__zzprivate_copy(void* to, const void* from, size_t bytes, cl_command_queue channel) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    khronos_opencl1_igpu__end t, f;
    if (s == 0 || !khronos_opencl1_igpu__zzprivate_end(s, to, &t) || !khronos_opencl1_igpu__zzprivate_end(s, from, &f)) return false;
    if (bytes == 0u) return true;
    const uint32_t owner = t.card ? t.owner : f.card ? f.owner : t.owner != 0xFFFFFFFFu ? t.owner : f.owner;
    cl_command_queue q = channel ? channel : owner < s->count ? s->queue[owner] : khronos_opencl1_igpu__zzpackage_queue();
    if (q == 0) return false;
    if (t.card && f.card) {
        if (t.owner != f.owner) return false;            /* two cards' memory are two contexts': never one copy */
        return clEnqueueCopyBuffer(q, f.buffer, t.buffer, f.offset, t.offset, bytes, 0u, 0, 0) == CL_SUCCESS &&
               clFinish(q) == CL_SUCCESS;
    }
    if (t.card) return clEnqueueWriteBuffer(q, t.buffer, CL_TRUE, t.offset, bytes, f.host, 0u, 0, 0) == CL_SUCCESS;
    if (f.card) return clEnqueueReadBuffer(q, f.buffer, CL_TRUE, f.offset, bytes, t.host, 0u, 0, 0) == CL_SUCCESS;
    if (!channel && clFinish(q) != CL_SUCCESS) return false;
    memcpy(t.host, f.host, bytes);
    return true;
}

/* ── THE DOORS ───────────────────────────────────────────────────────────────────────────────────────── */
static inline bool khronos_opencl1_igpu__memory_allocate(void** at, size_t bytes) {
    *at = khronos_opencl1_igpu__zzprivate_allocate(bytes, false, 0);
    return *at != 0;
}

static inline void khronos_opencl1_igpu__memory_free(void* at) { khronos_opencl1_igpu__zzprivate_release(at, false); }

static inline bool khronos_opencl1_igpu__memory_zerofill(void* at, size_t bytes) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    khronos_opencl1_igpu__end e;
    if (s == 0 || !khronos_opencl1_igpu__zzprivate_end(s, at, &e)) return false;
    cl_command_queue q = e.owner < s->count ? s->queue[e.owner] : khronos_opencl1_igpu__zzpackage_queue();
    if (q == 0) return false;
    if (bytes == 0u) return true;
    if (e.card) return khronos_opencl1_igpu__zzprivate_fill(q, e.buffer, e.offset, bytes);
    if (clFinish(q) != CL_SUCCESS) return false;
    memset(e.host, 0, bytes);
    return true;
}

static inline bool khronos_opencl1_igpu__memory_read(void* to, const void* from, size_t bytes) {
    return khronos_opencl1_igpu__zzprivate_copy(to, from, bytes, 0);
}

static inline bool khronos_opencl1_igpu__memory_write(void* to, const void* from, size_t bytes) {
    return khronos_opencl1_igpu__zzprivate_copy(to, from, bytes, 0);
}

/* The card's queue drained, and mapped RAM's newer card copies read back. */
static inline bool khronos_opencl1_igpu__compute_completed(void) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    cl_command_queue q = khronos_opencl1_igpu__zzpackage_queue();
    return q && clFinish(q) == CL_SUCCESS && khronos_opencl1_igpu__zzprivate_catch_up(s, khronos_opencl1_igpu__zzprivate_bound);
}

/* The side channel: a second queue on the same card, so a copy through it does not wait behind the work
 * in the card's own queue. */
static inline bool khronos_opencl1_igpu__side_open(void** channel) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    if (s == 0 || khronos_opencl1_igpu__zzprivate_bound >= s->count) return false;
    cl_int err = CL_SUCCESS;
    cl_command_queue q = clCreateCommandQueue(s->context[khronos_opencl1_igpu__zzprivate_bound],
                                              s->device[khronos_opencl1_igpu__zzprivate_bound], 0, &err);
    if (err != CL_SUCCESS) return false;
    *channel = (void*)q;
    return true;
}

static inline void khronos_opencl1_igpu__side_close(void* channel) {
    if (channel) (void)clReleaseCommandQueue((cl_command_queue)channel);
}

static inline bool khronos_opencl1_igpu__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    return khronos_opencl1_igpu__zzprivate_copy(to, from, bytes, (cl_command_queue)channel);
}

static inline bool khronos_opencl1_igpu__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    return khronos_opencl1_igpu__zzprivate_copy(to, from, bytes, (cl_command_queue)channel);
}

/* Mapped host RAM: `*at` the program's copy, `*card_at` the card's address. ▶ the note at the top. */
static inline bool khronos_opencl1_igpu__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    *at = 0;
    *card_at = khronos_opencl1_igpu__zzprivate_allocate(bytes, true, at);
    return *card_at != 0 && *at != 0;
}

static inline void khronos_opencl1_igpu__memory_free_host_ram(void* at) { khronos_opencl1_igpu__zzprivate_release(at, true); }

/* ⛔ OpenCL 1.2 cannot hand a kernel RAM it did not allocate, so this family answers no, and a caller copies. */
static inline bool khronos_opencl1_igpu__memory_register_host(void* at, size_t bytes, void** card_at) {
    (void)at; (void)bytes; *card_at = 0;
    return false;
}
static inline void khronos_opencl1_igpu__memory_unregister_host(void* at) { (void)at; }

static inline uint32_t khronos_opencl1_igpu__device_count(void) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    return s ? (uint32_t)s->count : 0u;
}

static inline uint32_t khronos_opencl1_igpu__bind_device(uint32_t index) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    if (s == 0 || index >= s->count) return 0u;
    khronos_opencl1_igpu__zzprivate_bound = index;
    return 1u;
}

static inline uint32_t khronos_opencl1_igpu__bound_device(void) {
    khronos_opencl1_igpu__state* s = khronos_opencl1_igpu__zzpackage_up();
    return (s && khronos_opencl1_igpu__zzprivate_bound < s->count) ? khronos_opencl1_igpu__zzprivate_bound : 0xFFFFFFFFu;
}

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. A card: through a bounce buffer and this family's own copy. */
static inline bool khronos_opencl1_igpu__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void khronos_opencl1_igpu__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory: a card's memory is not the machine's, so never here. */
static inline bool khronos_opencl1_igpu__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    (void)at; (void)bytes; (void)handle; (void)offset;
    return false;
}
/* A range of this family's memory is always in it: nothing is paged in, so nothing is waited for. */
static inline bool khronos_opencl1_igpu__memory_prefetch(const void* at, size_t bytes) {
    (void)at; (void)bytes;
    return true;
}
static inline bool khronos_opencl1_igpu__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read_through(handle, offset, bytes, to, khronos_opencl1_igpu__memory_write);
}

/* ── DIRECT ACCESS AND MARKS — not answered by this family: no device reads another's memory here, and a caller that
 * needs either refuses. */
static inline bool khronos_opencl1_igpu__peer_access(uint32_t other) { (void)other; return false; }
static inline bool khronos_opencl1_igpu__mark_open(void** mark) { (void)mark; return false; }
static inline void khronos_opencl1_igpu__mark_close(void* mark) { (void)mark; }
static inline bool khronos_opencl1_igpu__mark_record(void* mark) { (void)mark; return false; }
static inline bool khronos_opencl1_igpu__mark_wait(void* mark) { (void)mark; return false; }
/* No marks, so a queued copy is the side channel's own, done when the call returns. */
static inline bool khronos_opencl1_igpu__side_memory_write_queued(void* to, const void* from, size_t bytes, void* channel) {
    return khronos_opencl1_igpu__side_memory_write(to, from, bytes, channel);
}
static inline bool khronos_opencl1_igpu__side_memory_read_queued(void* to, const void* from, size_t bytes, void* channel) {
    return khronos_opencl1_igpu__side_memory_read(to, from, bytes, channel);
}
static inline bool khronos_opencl1_igpu__side_mark_record(void* mark, void* channel) { (void)mark; (void)channel; return false; }
static inline bool khronos_opencl1_igpu__mark_reached(void* mark) { (void)mark; return false; }

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_SYS_CUH */
