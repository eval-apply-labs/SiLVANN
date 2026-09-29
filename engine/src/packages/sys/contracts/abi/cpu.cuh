#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_ABI_CPU_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_ABI_CPU_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "../objects/heap_node.cuh"   /* the node `peek` hands back */

/* ══ ⭐⭐ sys's C INTERFACE — WHAT A PROGRAM OUTSIDE THE LANGUAGE MAY ASK OF sys ═════════════════════════
 *
 * ⚖ *"those do look like engine shaped things, but the looking inside do feel like they are sys … maybe
 * some things like the looking inside should be sys abi functions"* · *"heap-only … do it now"*. The engine
 * keeps the machine: standing it up, building and running programs, residency. What is left is `sys`'s
 * own: looking at the heap, and the allocator's and the fault channel's state. Python binds these as
 * `sys.<name>`.
 * ⛳ THE HEAP IS ON THE HOST, so every one of these is a plain read of this process' memory — no card, no
 * copy, and no waiting. A node read while blocks are writing it can be torn; `peek` promises a node, not
 * a moment. An address between two nodes is refused rather than rounded.
 * ⛳ HEAP-ONLY. An address outside the heap is refused: bytes in a package's buffers are that package's to
 * read, through its own interface (nn's `read`/`write`).
 * ⛳ ALL ANSWER 0 WHEN NOTHING IS UP, which is also what an address or a block that names nothing answers —
 * a caller checks one thing either way. */
extern "C" {
/* One node of the heap, by address. */
int                sys_abi_peek(unsigned long long address, sys__heap_node* into);
/* A heap offset as an address, or 0 for one that names nothing. */
unsigned long long sys_abi_full_address(unsigned long long offset);
/* Where block `block`'s computing base is, or 0 for a block the machine does not have. */
unsigned long long sys_abi_base_address(unsigned int block);
/* The allocator's tallies. `counts_deallocations` says whether the third is counted at all in this build. */
int                sys_abi_counts_deallocations(void);
int                sys_abi_counters(unsigned int* chunk_claims, unsigned int* chunk_refusals,
                                    unsigned int* deallocations);
/* The fault channel: how many raised, and the last raiser's pc and word; and putting it back to nothing. */
int                sys_abi_fault(unsigned int* raised, unsigned long long* failed_pc, unsigned long long* failed_op);
int                sys_abi_fault_clear(void);
}

/* ══ ⭐⭐ AND THE SAME INTERFACE AS DATA — WHAT A HOST LANGUAGE BINDS ═══════════════════════════════════
 *
 * ⚖ *"i would have kept them as function that return the content, like the pybind.cuh goes through the x
 * macro and calls the same method in each x macro id"* — and, of two readings, *"go with B"*: the content
 * is DATA. Every package answers `<pkg>_abi_surface()`: a table with one row per call a host may make, each
 * giving its name, its shape, the C function, the names of its arguments and answers, what a refusal is
 * said as, and a line of documentation. The binding (`pybind.cuh`) walks the package roster, asks each
 * package for its table and binds every row by its shape.
 * ⛳ SO A PACKAGE NAMES NO HOST LANGUAGE, and stays C: the table is plain data, and a binding for another
 *   language would read the same one. Adding a call is a row; a call of a new SHAPE is also one case in the
 *   binder, once, for every package.
 * ⛳ THE SHAPES ARE sys's because every package builds on sys, and one list of them is what lets one binder
 *   serve all. Each names the C signature a row's `call` has, and what the host is handed back: */
enum {
    SYS__ABI_SHAPE_NUMBER_TO_NUMBER = 1u,   /* unsigned long long f(unsigned long long)       -> a number        */
    SYS__ABI_SHAPE_ADDRESS_TO_NODE  = 2u,   /* int f(unsigned long long, sys__heap_node*)      -> (kind, op_code,
                                                                                                  [six args])    */
    SYS__ABI_SHAPE_WORDS            = 3u,   /* int f(unsigned long long* words, int* present)  -> {answer: number,
                                                                                                  or None where
                                                                                                  not present}   */
    SYS__ABI_SHAPE_NOTHING          = 4u,   /* int f(void)                                     -> nothing        */
    SYS__ABI_SHAPE_READ_BYTES       = 5u,   /* int f(unsigned long long address,
                                                     unsigned long long bytes, void* into)     -> the bytes      */
    SYS__ABI_SHAPE_WRITE_BYTES      = 6u    /* int f(unsigned long long address,
                                                     unsigned long long bytes, const void* from) -> how many   */
};
/* ⛳ An `int` answer of 0 is a REFUSAL, said as the row's `refused`; a row whose `refused` is null cannot
 *   refuse, and its answer is handed back as it is. ⛳ A WORDS row's `present` lets an answer be ABSENT — a
 *   tally compiled out is not a tally of zero, and the host is told the difference. */
#define SYS__ABI_MOST_WORDS  8u

typedef void (*sys__abi_call)(void);           /* a row's C function, called back through its shape's type */

typedef struct sys__abi_entry {
    const char*   name;                         /* as the host spells it: `peek`                          */
    uint32_t      shape;                        /* SYS__ABI_SHAPE_…                                       */
    sys__abi_call call;                         /* the C function, of the shape's signature               */
    const char*   args[2];                      /* its arguments' names, in order; null past the last     */
    const char*   answers[SYS__ABI_MOST_WORDS]; /* a WORDS row's answers, by name; null past the last     */
    const char*   refused;                      /* what a refusal is said as; null when it cannot refuse  */
    const char*   doc;                          /* one line for the host's help                           */
} sys__abi_entry;

typedef struct sys__abi_surface {
    const char*           about;                /* the package, in one line                               */
    uint32_t              count;
    const sys__abi_entry* entries;
} sys__abi_surface;

extern "C" {
/* sys's calls, as data. Null in a build with no heap to look at (a test harness). */
const sys__abi_surface* sys_abi_surface(void);
}

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_ABI_CPU_CUH */
