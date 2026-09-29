#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_ATOMIC_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_ATOMIC_CUH
#include "silicon__header.cuh"   /* the contract these primitives answer */
/* ══ THE SEAM'S ATOMICS, ON A CPU ════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"sys and the evaluator go host."* This is the seam's only backend, and it exists for the reason
 * the seam exists at all — ⚖ *"porting a backend means porting the subset a package needs."* The
 * evaluator-side subset is ten functions; these are six of them.
 *
 * ⛔⛔ AND THEY ARE WRITTEN, NOT PROMOTED FROM `test/src_host_shim.cuh`. The shim's versions look like
 * these and are NOT these: its `cas_u32` is a plain read-then-write carrying a `g_cas_interfere`
 * injection hook so a test can land a competitor inside the window on purpose, and its `block_id` is a
 * hardcoded `0`. Those are instruments. ⇒ ★★ A HARNESS STAND-IN IS WRITTEN TO BE *STEERABLE*, WHICH IS
 * THE OPPOSITE OF WHAT A SHIPPING PRIMITIVE IS FOR — lifting one into the engine would put a test's
 * back door on the shipping path and call it a backend.
 *
 * ⛳ THE ATOMICS ARE REAL BECAUSE THE HOST EVALUATOR IS NOT SINGLE-THREADED. ⚖ *"one thread
 * per gpu and one for the orchestrator at id 0."* The heap's chunk ring, the object counts and the
 * computing bases are shared across those threads exactly as they are across blocks on a card, so a relaxed
 * read-modify-write here is the same defect it would be on the card.
 * ⛳ `__atomic_*` RATHER THAN `<atomic>`: this header is included by C-subset code that must also
 * compile as C, and the builtins are what both front ends have. ▶ `src_c_subset_gate`. */

static inline unsigned int sys__silicon__cas_u32(unsigned int* address,
                                                  unsigned int compare, unsigned int value) {
    /* ⛳ STRONG, AND THE SEAM'S CONTRACT SAYS SO: it never fails when the word matched. The weak form
     * may fail spuriously, and every caller here loops on the RETURNED value rather than on a bool,
     * so a spurious failure would read as "somebody else won" and re-run work that nobody contested. */
    unsigned int expected = compare;
    __atomic_compare_exchange_n(address, &expected, value, /*weak=*/0,
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;                       /* the word as it was found, won or lost */
}

static inline unsigned int sys__silicon__add_u32(unsigned int* address, unsigned int addend) {
    return __atomic_fetch_add(address, addend, __ATOMIC_SEQ_CST);
}

static inline unsigned int sys__silicon__read_u32(const unsigned int* address) {
    /* ⛳ AN ATOMIC LOAD, NOT A PLAIN ONE. What this reads is written by another evaluator thread
     * without a lock, so a plain read is a data race the compiler is entitled to fold. */
    return __atomic_load_n((const volatile unsigned int*)address, __ATOMIC_SEQ_CST);
}

/* ⛳ A PAUSE, NOT A SLEEP. The callers are bounded spin loops — `sys__computing_base__read_result`
 * polls with this between tries — and a real sleep would turn a microsecond wait into a millisecond
 * one. `__builtin_ia32_pause` is the x86 spelling; anything else gets the compiler barrier, which is
 * still enough to stop the loop being hoisted. */
static inline void sys__silicon__wait_cycles(unsigned int n) {
    for (unsigned int i = 0u; i < n; ++i) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
}

/* ⛳ ON A CARD THESE ARE CACHE-SCOPE FENCES; ON A CPU THE SCOPES COLLAPSE INTO ONE. A host evaluator's
 * threads share one coherent memory, so "visible to my block" and "visible to the host" are the same
 * statement — and both are a full barrier rather than nothing, because the ORDERING still has to hold. */
static inline void sys__silicon__publish(void)         { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline void sys__silicon__publish_to_host(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_ATOMIC_CUH */
