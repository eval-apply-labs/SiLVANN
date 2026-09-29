#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__IMPL_CUH

/* ══ THE SILICON BACKEND SELECTOR ════════════════════════════════════════════════════════════════════
 *
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * What changed in this selector when the CPU arm arrived, so nobody restores the two-backend reading.
 *   · The top of this file once read *"THE BACKEND, AND ONLY ON A DEVICE COMPILE"*, *"the selector is
 *     `src_old/`'s, unchanged"* and *"NEITHER ARM ARRIVES ON A HOST BUILD"* — true of two backends, false
 *     of three. A porter reading it would have concluded there is no host selection path and added a
 *     second one.
 *   · The CPU arm once included the vendor's `hip/host.cuh`, reasoning that "the doors to the card are
 *     still the card's". It cannot: see the note on that arm.
 * ⛳ RETIREMENT: when nobody remembers the two-backend selector — at release, with the rest.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛔ THE BACKEND. `silicon__header.cuh` DECLARES every seam verb and defines none of them — that is the
 * whole point of a seam — so the arm below has to arrive or the package links against nothing. It pulls
 * in `cpu/silicon/` — CPU code, on a host compile. There is no other selection path, and a second one
 * would be two ways to choose a target.
 *
 * ⛳ A host HARNESS (`SYS__SILICON__HARNESS`) arrives with no arm, because `test/src_host_shim.cuh`
 * defines its own stand-ins before including the package and a real backend beside them would be a
 * redefinition.
 * ⛳ THAT COVERS THE HOST VERBS TOO, AND THEY NEED NO STAND-INS IN A HARNESS: it allocates nothing on a
 * device and launches nothing, so it calls none of them and has nothing to stand in for.
 * ⚠ A vendor's `host.cuh` compiles only against its vendor runtime, which this translation unit does not
 * have — which is exactly why the arm below cannot include it, and why there is a
 * `silicon_families/host/sys.cuh` at all.
 *
 * ⛳ AND THIS FILE DEFINES NO SEAM VERB ITSELF: it includes the card table and the arm's files, and
 * nothing else. A backend definition is compiled WITHOUT the seam's declaration in scope and cannot be
 * checked against it where it is written.
 * ⭐ IT IS CAUGHT ANYWAY, AND BY LINKAGE RATHER THAN BY DECLARATION. A drifted definition is a second
 * function the compiler is happy to have, but every CALLER was compiled against the declared signature,
 * and nothing defines that one — so the build ends at the link with an unresolved reference naming a
 * signature nobody wrote. `MEASURED` on the harness: 0 compile errors, no binary.
 * ⛔ THE CONDITION IS THAT SOMEBODY CALLS IT. A drifted verb with no caller leaves no unresolved
 * reference and links clean, which is why the seam's conformance is also checked by signature, caller or
 * no caller. ⇒ ★ A CHECK THAT ONLY FIRES ON WHAT IS EXERCISED IS BLIND EXACTLY WHERE A PORT IS THIN. */
/* ⭐⭐⭐ THE HOST BACKEND.
 * ⚖ *"sys and the evaluator go host."* The evaluator runs on the CPU, so the seam's DEVICE-SIDE
 * verbs — the atomics, the block identity and the scratch allocator —
 * have CPU implementations here.
 * ⛔ THE CARDS ARE REACHED THROUGH FAMILIES: `hipMalloc`, `hipMemcpy` and the streams live in a family
 * blob offered at runtime, never in this translation unit.
 * ⇒ ★★ "THE EVALUATOR ON THE HOST" IS NOT "NO DEVICE". The seam splits along which PROCESSOR RUNS THE
 *   CODE, not along which package it is in, and the two halves of the seam sit on opposite sides of
 *   that line. */
/* ⭐ THE TABLE FIRST — every silicon family offers itself into it, and the published `sys__gpu__` names
 * are its forwarders. ▶ `silicon_family.cuh`. */
#include "silicon_family.cuh"

#ifndef SYS__SILICON__HARNESS
    #include "atomic.cuh"
    #include "block.cuh"
    #include "memory.cuh"
    /* ⛔⛔ AND NO FAMILY. The host's doors — `malloc` as a card — are a silicon family like the others,
     * `silicon_families/host/`, and the program includes it after the packages. What is here is the
     * evaluator's own primitives, which a family never provides.
     * ⛔ A VENDOR'S FAMILY CANNOT BE INCLUDED IN THIS TRANSLATION UNIT AT ANY PRICE. Its `sys.cuh` opens
     *   with `#include <hip/hip_runtime.h>`, which cannot follow our `#define __device__` — the define
     *   poisons HIP's own headers, so **the host TU must be vendor-free**. `MEASURED`, a probe TU that
     *   included it, whose one and only error was:
     *       hip/host.cuh:7: fatal error: hip/hip_runtime.h: No such file or directory
     * ⇒ ★★ THE HOST TU CANNOT NAME A VENDOR, SO THE CARD IT TALKS TO CANNOT BE A HEADER — it arrives
     *   as a family's binary, offered at runtime. */
#endif

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__IMPL_CUH */
