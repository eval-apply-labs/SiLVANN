#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_ENVIRONMENT_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_ENVIRONMENT_CUH
/* ══ ⭐⭐⭐ THE DEVICE ENVIRONMENT, STOOD IN FOR — INCLUDE THIS FIRST, BEFORE ANYTHING ═════════════════
 *
 * ⚖ *"sys and the evaluator go host."*
 *
 * The rest of `cpu/silicon/` implements the seam's VERBS for a CPU. This file is different in kind: it
 * stands in for the device LANGUAGE — the keywords the tree is written in and a host compiler has never
 * heard of. It is the only file here that must be included before `packages/manifest.cuh`, because it is
 * what makes `__device__` mean nothing; included after, every declaration in the tree is a wall of
 * *"'__device__' does not name a type"*.
 *
 * ⛳ **AND IT IS SHORT BECAUSE THE SEAM ALREADY DID THE WORK.** `MEASURED` across all of
 * `sys` + `engine`, and `threadIdx`: `threadIdx` **0** · `blockIdx` **0** · `blockDim` **0** · `gridDim` **0** · `__syncthreads` **0** ·
 * `atomicAdd`/`atomicOr` **0** · `__shared__` **1**, and that one is `silicon__header.cuh`'s own
 * `BLOCK_LOCAL` default. Every one of those apparent hits is a COMMENT. ⇒ ★ A TREE THAT ASKS THE SEAM FOR ITS BLOCK ID
 * DOES NOT HAVE TO BE TAUGHT THAT THERE IS NO GRID — it never asked about one.
 *
 * ⛔⛔ **THIS IS NOT `test/src_host_shim.cuh` AND MUST NOT BECOME IT.** That file compiles the same tree
 * under `g++` for the host suite, and it is a HARNESS: its `cas_u32` is a plain read-then-write
 * with a test-injection hook, and its `BLOCK_LOCAL` is `static` because it runs one block. Both are
 * correct there and catastrophic here — `MEASURED` in `test/src_host_silicon_selftest.cpp`, a plain
 * read-modify-write loses **87.5% of 3.2M increments** under eight real threads. ⇒ ★★ THE STAND-IN THAT
 * CANNOT BE WRONG IN A TEST IS ALWAYS WRONG IN PRODUCTION. This file supplies only the KEYWORDS; every
 * operation comes from `cpu/silicon/{atomic,block,memory}.cuh`, which are the real thing.
 */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 *   · The `__noinline__` note below once cited *"`MEASURED` at −2.04% and 217 fewer spilled registers"*.
 *     Those are `k_eval` figures (S98, fib(15)); the evaluator has left the card and they do not
 *     transfer. The cold-path argument for keeping `raise` out of line stands on its own.
 * ⛳ RETIREMENT: at release, with the rest.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── the keywords ────────────────────────────────────────────────────────────────────────────────── */
#define __device__
#define __global__
/* ⛳ A LAUNCH GEOMETRY HINT WITH NO LAUNCH TO HINT AT. `ENG_BOOT_BLOCK` expands to this and is written
 * on every kernel; it tells a device compiler how many registers it may spend, which is not a sentence
 * about anything here. */
#define __launch_bounds__(...)
/* ⛔ AND THIS ONE IS DEFINED RATHER THAN DEFINED AWAY, because deleting it would silently drop the one
 * property it exists to assert. `sys__fault__raise` is `__noinline__` on purpose — it is a cold path,
 * `REASONED` at `fault__impl.cuh` — and the host attribute means the same thing, so the function it
 * marks stays out of line here too. */
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif

/* ── ⭐⭐⭐ ONE COPY PER RUNNER, AND THIS IS THE LINE THAT DECIDES WHETHER FOUR CARDS RUN AS FOUR ──────
 * `silicon__header.cuh` defaults `BLOCK_LOCAL` to `static __shared__` — on a card, one copy per BLOCK.
 * The host analogue of "per block" is "per RUNNER THREAD", which is `__thread` and is not `static`.
 * ⛔⛔ `static` ALONE COMPILES, PASSES EVERY SINGLE-RUNNER TEST, AND IS THE BUG. Every evaluator thread
 * would share one computing base; the claim protocol would serialise them perfectly and silently, and a
 * four-card machine would do the work of one while looking healthy. There is no failure to observe —
 * only a number that is never as large as it should be.
 * ⛳ Tenants: `bindings__impl` · `eval.cuh` · `heap__impl` ×2 — the evaluator's own state. */
#define SYS__SILICON__BLOCK_LOCAL static __thread

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_ENVIRONMENT_CUH */
