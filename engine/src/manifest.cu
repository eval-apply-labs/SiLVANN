/* ══ src/manifest.cu — THE ONE TRANSLATION UNIT OF THE NESTED-FRAMES ENGINE ═══════════════════════════
 * ⛳ INVARIANT: `find src -name '*.cu' | wc -l` == 1.
 *   A second `.cu` means someone made a second TU by accident. `.cuh` is for everything included.
 *   It is also load-bearing for §3 ③ of the spec: ONE TU is why the compiler already has whole-call-graph
 *   visibility, which is why `static` measured as convention rather than mechanism.
 *
 * ⚖ SPEC: docs/nested_frames_engine_spec.md. Read §1 before changing anything here — the rework is
 *   justified by ONTOLOGY, not by registers, and §1 says so with the numbers that refute the alternative.
 *
 * BUILD:  bash scripts/rebuild_engine.sh
 * This unit is `manifest.cu` to `engine/manifest.cuh` to `packages/manifest.cuh` to `sys` and stops; the
 *   harness includes the same two and nothing else.
 * ⛳ THE ONE THING THAT LOOKS LIKE A COUPLING IS A GUARD WITH NO WRITER. `packages/manifest__header.cuh`
 *   wraps its package-id block in `#ifndef SILVANN_PACKAGE_IDS_PRESENT`, against a second registry that
 *   also names a package `sys` at id zero — a `constexpr` is not a macro, so two identical spellings are
 *   a redefinition rather than an agreement. Nothing this build can reach sets that flag, so the branch
 *   is always taken, and that file says so beside the guard: its subject is COMPOSITION, so it is
 *   already subjectless and goes whenever the line that would set the flag goes. */

/* ⭐⭐⭐ THE PROLOGUE, AND IT MUST COME BEFORE EVERYTHING — ⚖ *"sys and the evaluator go
 * host"*. `environment.cuh` is what defines `__device__` away; anything included ahead of it meets a
 * wall of *"'__device__' does not name a type"*, which is exactly what this TU did the first time g++
 * was pointed at it: 880 of those and 34,719 "stray" errors behind them, all from one missing include.
 * ⛳ AND THE std HEADERS COME WITH IT because a device compile gets them from the vendor's runtime and a
 * host compile has no such benefactor — the first error of all was `uint64_t` not naming a type. */
#include "packages/sys/cpu/silicon/environment.cuh"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "engine/manifest.cuh"
/* ⛳ AND THE HOST BOUNDARY, WHICH IS A FILE AND NOT A BLOCK IN THIS ONE. ⚖ ARCHITECT: *"i think we
 * should have had a pybind.cuh file to hold this instead of letting it into the main manifest, a bit
 * like we did with the old src."* Right, and the older tree is the precedent: its module body is two
 * macro expansions and every binding definition lives with whatever owns it.
 * ⇒ ★ THIS FILE LISTS WHAT THE BUILD CONTAINS. A definition written here is a thing whose owner nobody
 *   had to name, and the reason it ends up here is that this is where the entry point already was.
 * ⛳ WHY A FILE OF ITS OWN RATHER THAN A PACKAGE'S REGISTRAR, which is what the older tree uses: over
 *   there a binding belongs to a PACKAGE and each publishes one. Here the bindings are over the
 *   ENGINE'S C interface, so there is no package to own them — and the engine may not carry a binding
 *   symbol at all. What is left is the boundary itself, which is what this file is. */
#include "pybind.cuh"
