#ifndef SILVANN__ENGINE_ABI_CUH
#define SILVANN__ENGINE_ABI_CUH
/* ══ THE C INTERFACE ═══════════════════════════════════════════════════════════════════════════════════
 *
 * The outermost point of the engine: what a caller outside this language can name.
 *
 * ⭐⭐ WHY THERE IS A TIER HERE AT ALL, AND NOT JUST A BINDING. Something has to stand between a
 *   caller and the engine, and what it is shaped like decides what can ever reach in. A C interface is
 *   the narrowest useful shape: no templates, no classes, no exceptions, no ownership that has to be
 *   explained. Anything that can call C can call this, and a binding for one particular language
 *   becomes a thin thing written against it rather than the only door.
 * ⛳ THE DOORS ARE HOST CALLS. Every one reaches the machine through `ENG__ABI__ON_THE_MACHINE` (the
 *   caller's own thread as runner zero), `ENG__LAUNCH_MANY` (a runner per block, joined before it returns)
 *   or `ENG__LAUNCH_RESIDENT` (runners left running) — ▶ `engine/launch.cuh`.
 *
 * ⛳ ONE MACHINE, HELD IN A STATIC, AND THAT IS HONEST RATHER THAN LAZY. A handle per machine would cost
 *   nothing to write and would mean nothing to use: a machine is a pool and a register, and everything
 *   below reaches them through statics this process holds one of — so a second machine is not a thing this
 *   process can have, however many blocks the one it has seats. A handle is what this grows when that
 *   changes, and the shape of every call below is unaffected by it — each already takes what it works on.
 *
 * ⛳ WHAT A PROGRAM COSTS TO BUILD, SAID PLAINLY. Every cell is one call through the launch seam, which
 *   on the host is a plain call — no settle, no copy back. If a composer emitting thousands ever makes
 *   that the wrong shape, the answer is a tape: hand the whole description over in one buffer and let a
 *   single call walk it. Nothing here has to change for that; it becomes one more entry point beside
 *   these. ⛔ The per-cell cost a tape would save is UNMEASURED on the host — measure it before building one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ⛔⛔ THIS GUARD HIDES EVERYTHING TO THE `#endif` AT THE END OF THE FILE: every file of `abi/`, and with
 * them every `extern "C"` door pybind binds. A harness build (`SYS__SILICON__HARNESS`) gets none of this
 * file, and a probe TU compiled against it reports **0 errors** having compiled nothing.
 * ⇒ ★★ A CLEAN COMPILE OF A GUARDED-OUT FILE IS INDISTINGUISHABLE FROM A CLEAN COMPILE. When a probe
 *   comes back suspiciously clean, check which side of `SYS__SILICON__HARNESS` it compiled before
 *   believing it. */
#ifndef SYS__SILICON__HARNESS   /* a real build, not a test harness that brings its own silicon */
#include "abi/machine.cuh"
#include "abi/building.cuh"
#include "abi/running.cuh"
#include "abi/residency.cuh"
#include "abi/silicon.cuh"
#include "abi/vocabulary.cuh"

#endif  /* not the harness */

#endif /* SILVANN__ENGINE_ABI_CUH */
