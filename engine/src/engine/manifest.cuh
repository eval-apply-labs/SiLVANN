#ifndef SILVANN__ENGINE_MANIFEST_CUH
#define SILVANN__ENGINE_MANIFEST_CUH
/* ══ THE EVAL-APPLY GPU INTERPRETER — INCLUDE ORDER IS THE DEPENDENCY DIRECTION ═══════════════════════
 * ⭐⭐ THREE TIERS, AND ONE AXIS DECIDES WHICH A THING BELONGS TO:
 *       *"the separation is between a lisp interpreter and a library to compute neural networks."*
 *       `sys__`        the LANGUAGE you write IN   — list ADT, symbols, opcodes, types, silicon
 *       `nn__`         the LIBRARY you build WITH  — the model ops
 *       `eng__`        THE INTERPRETER             — eval/apply, and what a block does between programs
 *   They are name prefixes and not namespaces: the tree is written in a C subset, which has none, and the
 *   C-subset gate refuses one. The packages are the rows of `PACKAGE_LIST` in
 *   `packages/manifest__header.cuh` — `sys` and `nn` — and `packages/manifest.cuh` is generated from it.
 *
 * ⭐ THE TRANSLATION UNIT REACHES NOTHING OUTSIDE `src/` BUT THE TOOLCHAIN'S OWN HEADERS. The claim gate's
 *   `src_reaches_only_itself` and `no_borrow_from_src` rules check it (`python3 scripts/src_claim_gate.py`
 *   from `backstage/`). ⛔ AND NOTHING COUPLES THIS TREE TO `src_old/`, SO THE COMPILER IS NOT A GUARD
 *   HERE: an edit made over there reaches nothing in this build. The language this engine runs on is
 *   `packages/sys/`, and that is the folder to open.
 * ⛳ THE DECOUPLING IS CHEAP TO LOSE: one include of the other tree's manifest puts the WHOLE of it — its
 *   router and every package behind it — into this single translation unit, to borrow a handful of names.
 *   ⇒ ★ BORROW A NAME, TAKE A TREE.
 * ⛳ THE `sys__`/`SYS__` PREFIXING STANDS ON ITS OWN ARGUMENT — ⚖ *"sys__ remains for clarity"*: an id
 *   separates what the MACHINE reads, and a NAME is the only thing that separates what a person writes, so
 *   two packages both publishing `clone` would collide in the one place the packing never touches. The
 *   reasoning is in `packages/sys/language_contract.cuh`.
 *
 * ⛳ WHAT THE TRANSLATION UNIT HOLDS, IN ORDER: the packages, the silicon families, then the engine's own
 *   files. The packages go FIRST because everything below them is written against the language — the
 *   evaluator refuses through `sys__fault__raise`, `boot.cuh` takes the room for the fault channel and
 *   publishes it, and `abi.cuh` reads it back.
 * ⛳ AND THE FAULT CHANNEL IS THE PACKAGE'S OWN OBJECT: `sys__fault_channel` is three words — a count, a
 *   pc and an op — declared in `packages/sys/contracts/objects/fault.cuh`. */
#include "../packages/manifest.cuh"   /* the packages: what a program can name and what it is made of */

/* ⭐ AND THE SILICON FAMILIES, which answer what the packages ask — after them, because a family is written
 * against their lists, and before the engine, which stands a machine up on one. Through the tier's own
 * manifest, which knows what the program takes of it. ▶ `silicon_families/manifest.cuh`. */
#include "../silicon_families/manifest.cuh"

#include "launch.cuh"        /* how a kernel is started — a call, or a thread pool. ▶ the file for why */
#include "bindings.cuh"      /* why the environment is ONE object — it is `sys__bindings`           */
#include "eval.cuh"          /* the evaluator: a flat scan, and apply is the call of the opcodes in it */
#include "opcodes.cuh"       /* and what a block does between programs: wait for one, run it, say so  */
#include "boot.cuh"          /* standing the machine up: memory, a published allocator, a started block */
#include "abi.cuh"           /* what a caller outside this language can name                        */

#endif /* SILVANN__ENGINE_MANIFEST_CUH */
