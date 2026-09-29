#ifndef SILVANN__PACKAGES_SYS_MANIFEST__HEADER_CUH
#define SILVANN__PACKAGES_SYS_MANIFEST__HEADER_CUH
/* ══ sys — WHAT THE PACKAGE DECLARES ═════════════════════════════════════════════════════════════════
 *
 * ⭐ THIS IS A LIST AND NOT AN ORDER, and that is the whole of what it is for. Every file names what it
 * needs at the top of itself, so the compiler works the order out and getting this list wrong in that
 * respect is not possible. `MEASURED`: shuffling these lines five times over gives five builds and
 * five identical passes.
 *
 * ⛳ SO WHY A LIST AT ALL. Headers arrive on their own — anything that needs one includes it. An
 * IMPLEMENTATION does not: nothing includes an impl, because nothing needs a definition, only a
 * declaration. Something therefore has to NAME them or they are never compiled. That is this file and
 * its sibling: what the package DECLARES here, what it DEFINES in `manifest.cuh`. Adding a file is
 * appending a line to whichever of the two it belongs in.
 *
 * ⛳ AND A FILE THAT IS NOT SPLIT HAS NO PHASE OF ITS OWN. The files below that carry no `__header` or
 * `__impl` in their name are listed here because this is where they are first needed — and because a
 * file every reader arrives at through somebody else's include is one the list only has to make sure is
 * COMPILED, not one it has to place.
 * ⛔ NO COUNT HERE — a number in that sentence goes stale with the next include. **RE-DERIVE:**
 * `grep -oE '#include "[a-z_]+\.cuh"' packages/sys/manifest__header.cuh | grep -vE '__header|__impl'`.
 *
 * ── WHY THE FILES INCLUDE THEMSELVES RATHER THAN BEING ORDERED HERE ──────────────────────────────────
 * ⚖ ARCHITECT: *"it makes it easier to explicitly name the contracts at the point of reading, plus it
 * likely helps IDEs reference things."* Both, and a third that only showed up in the doing: an ordering
 * that lives in one file is a fact no file can be wrong about ON ITS OWN, so nothing local ever fails —
 * it fails somewhere else, later, in whatever was unlucky enough to be first.
 *
 * ⛳ WHAT MADE IT POSSIBLE, and it was not obvious before the headers existed: a call needs a
 * DECLARATION and never a definition, so once every file's contract was separated out, no implementation
 * had any reason to precede another. One did — a macro in the allocator that the object interface used —
 * and it became a declared function, which is the same thing said properly.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "contracts/objects/kind.cuh"
#include "cpu/heap_node__header.cuh"
#include "language_contract.cuh"
#include "cpu/heap_object__header.cuh"
#include "cpu/bindings__header.cuh"
#include "cpu/carousel__header.cuh"
#include "cpu/computing_base__header.cuh"
#include "cpu/dictionary.cuh"
#include "cpu/error.cuh"
#include "cpu/fault__header.cuh"
#include "cpu/heap__header.cuh"
#include "cpu/list__header.cuh"
#include "cpu/node_array__header.cuh"
#include "cpu/procedure.cuh"
#include "cpu/opcodes/opcodes.cuh"
#include "cpu/package__header.cuh"
#include "cpu/silicon/silicon__header.cuh"
#include "cpu/settings.cuh"
#include "cpu/stack__header.cuh"
#include "cpu/string.cuh"
#include "cpu/symbol.cuh"
#include "cpu/system_register__header.cuh"
#include "cpu/todo.cuh"
#include "contracts/defaults.cuh"
#include "cpu/opcodes/verb_abi.cuh"            /* the ruled verb boundary + the one bridge */
#include "cpu/opcodes/opcodes_abi__header.cuh" /* the verbs' ABI adapters, declared */

/* The package answering the roll call the registry takes after this phase. A row with no include says so
 * at the registry, naming this package, instead of somewhere further down. */
#define PACKAGE_sys_HEADER_PRESENT 1

#endif /* SILVANN__PACKAGES_SYS_MANIFEST__HEADER_CUH */
