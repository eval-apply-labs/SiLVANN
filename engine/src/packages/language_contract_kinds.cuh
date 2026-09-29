#ifndef SILVANN__PACKAGES_LANGUAGE_CONTRACT_KINDS_CUH
#define SILVANN__PACKAGES_LANGUAGE_CONTRACT_KINDS_CUH
/* ══ THE KINDS, NAMED — GENERATED FROM EVERY PACKAGE'S ROWS ══════════════════════════════════════════
 *
 * A kind is the first word of every node, and what a package registers is a number. These are the
 * constants that number is reached by, generated from the same rows the host's type table is, so a kind
 * cannot be called one thing in the tree and published as another.
 *
 * ⭐ THE VALUE IS THE PACKED KIND AND NOT THE ROW'S BASE. A row carries a base its package numbers from
 * zero; the package is applied here, once, at the gather. That arrangement is the tree's everywhere and
 * the reason is measured rather than preferred: applying a tag per row instead of once at a gather let
 * thirty-three opcodes escape untagged one category over — a collision inside the mechanism built to
 * prevent collisions, and silent, because the dispatch matched on names.
 *
 * ⭐⭐ AND THIS RUNS BEFORE ANY PACKAGE'S HEADERS, WHICH IS THE ONE THING THAT MAKES IT DIFFERENT FROM
 * THE VERBS NEXT DOOR. A verb constant is needed late — the dispatch that compares one is an
 * implementation, and every implementation is read after every declaration. A KIND is needed early: the
 * object interface asks whether a node is a reference by comparing its kind, the register answers a row
 * that holds nothing with one, and both of those are declarations that carry their bodies with them. So
 * the kinds are gathered in a phase of their own, ahead of the headers that compare them, and the verbs
 * stay where they are.
 * ⛳ WHAT MAKES THAT PHASE POSSIBLE IS THAT A CONTRACT FILE NEEDS NOTHING. It defines macros and expands
 * none, so every package's rows can be read before anything else in the tree has been.
 *
 * ⛳ AND IT SITS BESIDE THE ROSTER RATHER THAN INSIDE A PACKAGE, for the reason everything roster-wide
 * does: a gather reads EVERY package's rows, so it can only run once every package has written them, and
 * a package is read before some of the others. Put it inside one and the packages listed after it have
 * not spoken yet — the gather pastes a macro nobody has defined, and what the compiler reports is a
 * syntax error inside whichever package happened to be next, pages from the row that caused it.
 *
 * ⛳ THEY ARE `constexpr uint32_t` AND NOT AN ENUM. A second package's kinds carry its id at bit 24 and
 * fall outside the range an enum picks for its own members, which a device compiler refuses outright
 * while the host compiler accepts it silently. Stating an underlying type answers that only while
 * one package writes every row; a constant answers it for everybody.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include <stdint.h>

#include "manifest__header.cuh"

#define PACKAGE_KIND_ID(PKG, BASE, SPELLING, NAME)                                                      \
    constexpr uint32_t NAME = PACKAGE_DTYPE(PKG, BASE);
#define PACKAGE_KIND_ID_GATHER(NAME)       NAME##__LANGUAGE_CONTRACT__KINDS(PACKAGE_KIND_ID, NAME)
#define PACKAGE_KIND_ID_GATHER_ROW(ID, NAME, ...)   PACKAGE_KIND_ID_GATHER(NAME)
PACKAGE_LIST(PACKAGE_KIND_ID_GATHER_ROW)
#undef PACKAGE_KIND_ID
#undef PACKAGE_KIND_ID_GATHER
#undef PACKAGE_KIND_ID_GATHER_ROW

#endif /* SILVANN__PACKAGES_LANGUAGE_CONTRACT_KINDS_CUH */
