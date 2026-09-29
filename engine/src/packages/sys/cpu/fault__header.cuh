#ifndef SILVANN__PACKAGES_SYS_CPU_FAULT__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_FAULT__HEADER_CUH
/* ══ THE FAULT CHANNEL, DECLARED ══════════════════════════════════════════════════════════════════════
 *
 * How this package says a thing went wrong. Every verb that refuses raises before it returns, and the
 * word it raises with is its own — `SYS__<subject>__FAULT_<what>`, four ASCII characters so a reader
 * meets the reason and not a number.
 *
 * ⭐ IT IS A SEAM AND NOT AN IMPLEMENTATION, the same arrangement `silicon__header.cuh` uses and for the
 * same reason: WHERE a fault goes is a property of what the package is running inside. In a real build
 * it lands in three words the boot owns and the host reads through `sys_abi_fault`; in a test it lands
 * in two variables the harness asserts on. This package cares that it was raised and never where it went.
 *
 * ⛳ AND `pc` IS PASSED AS ZERO EVERYWHERE HERE, WHICH IS HONEST RATHER THAN LAZY. NOTHING IN THIS TREE
 * HAS A PROGRAM COUNTER TO HAND IT — the evaluator walks a form held in the heap rather than an
 * instruction stream, and a container verb is called from many sites and could not name the one it was
 * serving even if there were an address to name. `MEASURED`, by a command that excludes its own mention:
 * `grep -rn 'sys__fault__raise(' src | grep -v '(0ull' | grep -v 'fault__'` comes back empty.
 * ⇒ `op` carries the identity alone, writing zero says "not applicable" in the one place a reader would
 * look, and the slot stays open for a raiser that knows which form it was serving — the host reads it
 * through `sys_abi_fault` whatever is in it.
 *
 * ⛔ A FAULT IS NOT AN ERROR OBJECT. `error.cuh` builds a VALUE a program can hold, compare and return.
 * This is a side channel for the case where there is nobody to return to — the two are separate on
 * purpose and neither is a fallback for the other.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛳ NOT INLINED, AND THE REASON IS AT THE FILLING (`fault__impl.cuh`). A seam's shape is the
 * filling's business everywhere else in this package; this is the one property of it the DECLARATION has
 * to carry, because a caller that inlined the verb and a caller that called it would not agree about
 * what they were compiling. */
static __device__ __noinline__ void sys__fault__raise(uint64_t failed_pc, uint64_t failed_op);

#endif /* SILVANN__PACKAGES_SYS_CPU_FAULT__HEADER_CUH */
