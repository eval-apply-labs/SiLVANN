#ifndef SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "c_fixed_stack__header.cuh"
#include "silicon/silicon__header.cuh"
/* ══ THE C FIXED STACK, DEFINED ══════════════════════════════════════════════════════════════════════
 * The contract is in `c_fixed_stack__header.cuh`. This file owes nothing to anything else in the
 * package: a compare-and-swap, an add, a wait and room from the silicon seam, and the fault channel to say
 * no on.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Which thread is asking, as a number no other live thread answers, so the lock word can say who holds
 * it. `sys__silicon__block_id` cannot: it names the runner a thread serves, and two threads can serve the
 * same one — the caller's own thread becomes runner 0 for a launch of one, a launch that finds the pool
 * busy starts threads numbered from 0 again, and a thread that never became a runner reads 0. So a
 * thread draws its number the first time it asks, from a counter every thread shares, and keeps it.
 * Zero is never drawn, because zero is the free lock. The counter wraps after 2^32 - 1 draws, and only
 * then can a thread draw a number another thread still holds. */
static __device__ unsigned int sys__c_fixed_stack__zzprivate_tokens_drawn = 0u;
static __thread   unsigned int sys__c_fixed_stack__zzprivate_token        = 0u;

static __device__ inline unsigned int sys__c_fixed_stack__zzprivate_thread_token(void) {
    unsigned int token = sys__c_fixed_stack__zzprivate_token;
    if (token == 0u) {
        do token = sys__silicon__add_u32(&sys__c_fixed_stack__zzprivate_tokens_drawn, 1u) + 1u;
        while (token == 0u);
        sys__c_fixed_stack__zzprivate_token = token;
    }
    return token;
}

/* Take the lock, writing `me`, the asking thread's token, into its word. Answers zero when it is had, or
 * the fault word that says why not: HELD at once when the word already names `me`, since waiting would be
 * waiting on itself, and LOCK after `LOCK_TRIES` misses. The caller raises it, so each refusal raises
 * where it returns. */
static __device__ inline uint64_t sys__c_fixed_stack__zzprivate_lock(sys__c_fixed_stack* stack, unsigned int me) {
    for (unsigned int t = 0u; t < SYS__C_FIXED_STACK__LOCK_TRIES; ++t) {
        const unsigned int holder = sys__silicon__cas_u32(&stack->lock, 0u, me);
        if (holder == 0u) return 0ull;
        if (holder == me) return SYS__C_FIXED_STACK__FAULT_HELD;
        sys__silicon__wait_cycles(SYS__C_FIXED_STACK__LOCK_BACKOFF_CYCLES + sys__silicon__block_id());
    }
    return SYS__C_FIXED_STACK__FAULT_LOCK;
}

/* Give it back. A swap from `me`, so a thread giving back a lock it does not hold, nobody's or another
 * thread's, is caught, and the word is left naming whoever holds it. */
static __device__ inline void sys__c_fixed_stack__zzprivate_unlock(sys__c_fixed_stack* stack, unsigned int me) {
    if (sys__silicon__cas_u32(&stack->lock, me, 0u) != me)
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_LOCK);
}

/* How many values are in it. One aligned word read atomically, so it was true when it was read. */
static __device__ inline unsigned int sys__c_fixed_stack__count(const sys__c_fixed_stack* stack) {
    return (stack == 0) ? 0u : sys__silicon__read_u32(&stack->top);
}

/* Put one on top. */
static __device__ inline bool sys__c_fixed_stack__push(sys__c_fixed_stack* stack, uint32_t value) {
    if (stack == 0 || stack->live != SYS__C_FIXED_STACK__LIVE) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_NO_STACK);
        return false;
    }
    const unsigned int me = sys__c_fixed_stack__zzprivate_thread_token();
    const uint64_t refused = sys__c_fixed_stack__zzprivate_lock(stack, me);
    if (refused != 0ull) {
        sys__fault__raise(0ull, refused);
        return false;
    }
    if (stack->top >= stack->capacity) {
        sys__c_fixed_stack__zzprivate_unlock(stack, me);
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_FULL);
        return false;
    }
    stack->value[stack->top] = value;
    stack->top += 1u;
    sys__c_fixed_stack__zzprivate_unlock(stack, me);
    return true;
}

/* Take the top one, or answer false when there is nothing in it. */
static __device__ inline bool sys__c_fixed_stack__pop(sys__c_fixed_stack* stack, uint32_t* out) {
    if (stack == 0 || stack->live != SYS__C_FIXED_STACK__LIVE || out == 0) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_NO_STACK);
        return false;
    }
    const unsigned int me = sys__c_fixed_stack__zzprivate_thread_token();
    const uint64_t refused = sys__c_fixed_stack__zzprivate_lock(stack, me);
    if (refused != 0ull) {
        sys__fault__raise(0ull, refused);
        return false;
    }
    if (stack->top == 0u) {
        sys__c_fixed_stack__zzprivate_unlock(stack, me);
        return false;                                     /* empty: the one quiet no */
    }
    stack->top -= 1u;
    *out = stack->value[stack->top];
    sys__c_fixed_stack__zzprivate_unlock(stack, me);
    return true;
}

/* Ask the platform for room, lay the stack at the front of it and its values after, and answer that
 * address. One allocation, so the address `create` answers is the one `dispose` frees. The structure is
 * a whole number of words long, so the values after it are aligned. Nothing is zeroed: a slot is read
 * only below `top`, after a push has written it. */
static __device__ inline sys__c_fixed_stack* sys__c_fixed_stack__create(unsigned int capacity) {
    if (capacity == 0u) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_SIZE);
        return (sys__c_fixed_stack*)0;
    }
    /* A room that cannot be counted in the width the seam takes would wrap, and a room smaller than asked
     * for reads as success until the first write past its end. */
    const unsigned int header = (unsigned int)sizeof(sys__c_fixed_stack);
    if (capacity > (0xFFFFFFFFu - header) / (unsigned int)sizeof(uint32_t)) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_TOO_BIG);
        return (sys__c_fixed_stack*)0;
    }
    void* room = sys__silicon__alloc(header + capacity * (unsigned int)sizeof(uint32_t));
    if (room == 0) {
        /* The seam answers NULL rather than raising, because only a caller knows whether it can carry on
         * without the room. This one cannot. */
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_NO_ROOM);
        return (sys__c_fixed_stack*)0;
    }
    sys__c_fixed_stack* stack = (sys__c_fixed_stack*)room;
    stack->live     = SYS__C_FIXED_STACK__LIVE;
    stack->lock     = 0u;
    stack->capacity = capacity;
    stack->top      = 0u;
    stack->value    = (uint32_t*)((unsigned char*)room + header);
    return stack;
}

/* Give the room back, through the seam it came from. A stack with values in it is refused and keeps its
 * room, because an empty container and a full one are the same size and nothing about freeing one
 * notices the difference. */
static __device__ inline void sys__c_fixed_stack__dispose(sys__c_fixed_stack* stack) {
    if (stack == 0 || stack->live != SYS__C_FIXED_STACK__LIVE) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_NO_STACK);
        return;
    }
    if (sys__c_fixed_stack__count(stack) != 0u) {
        sys__fault__raise(0ull, SYS__C_FIXED_STACK__FAULT_NOT_EMPTY);
        return;
    }
    stack->live  = 0u;                       /* before the free, so the witness is gone either way */
    stack->value = 0;
    sys__silicon__free((void*)stack);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__IMPL_CUH */
