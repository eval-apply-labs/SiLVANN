#ifndef SILVANN__PACKAGES_LANGUAGE_CONTRACT_VERBS_CUH
#define SILVANN__PACKAGES_LANGUAGE_CONTRACT_VERBS_CUH
/* ══ THE VERBS, NAMED — GENERATED FROM EVERY PACKAGE'S ROWS ══════════════════════════════════════════
 *
 * A published verb has a number. Without this file it has only a number, and every caller — a composer,
 * a test, a person reading a dispatch — carries the mapping itself. These are the names, generated from
 * the same rows the dispatch is, so a verb cannot be named one thing and dispatched as another.
 *
 * ⭐ WHY IT SITS HERE AND NOT IN A PACKAGE, WHICH IS THE WHOLE REASON THE FILE EXISTS. The gather reads
 * EVERY package's rows, so it can only run once every package has written them — and a package is read
 * before some and after others. Put it inside one and the packages listed after that one have not been
 * seen yet: the gather pastes a macro name that is not defined, and what the compiler reports is a
 * syntax error inside the innocent package, pages away from the row that caused it.
 * ⇒ SO THE RULE IS THE ONE THE ROSTER ALREADY KEEPS: anything that reads all the packages belongs to
 * nobody and lives beside the list. A package reads its own rows; the container reads everybody's.
 *
 * ⭐ THE VALUE IS THE PACKED ID AND NOT THE ROW'S BASE, because the packed id is what a program holds and
 * what the switch compares. A row carries a base; the package is applied here, once, at the gather.
 * ⛳ AND THEY ARE `constexpr uint64_t` RATHER THAN AN ENUM, because a second package's verbs carry its id
 * at bit 32 and would not fit the type a plain enum picks for small values.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "manifest__header.cuh"

#define PACKAGE_VERB_ID(PKG, BASE, SPELLING, APPLY, NAME, ...)                                          \
    constexpr uint64_t NAME = PACKAGE_VERB(PKG, BASE);
#define PACKAGE_VERB_ID_GATHER(NAME)       NAME##__LANGUAGE_CONTRACT__VERBS(PACKAGE_VERB_ID, NAME)
#define PACKAGE_VERB_ID_GATHER_ROW(ID, NAME, ...)   PACKAGE_VERB_ID_GATHER(NAME)
PACKAGE_LIST(PACKAGE_VERB_ID_GATHER_ROW)
#undef PACKAGE_VERB_ID
#undef PACKAGE_VERB_ID_GATHER
#undef PACKAGE_VERB_ID_GATHER_ROW

#endif /* SILVANN__PACKAGES_LANGUAGE_CONTRACT_VERBS_CUH */
