#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_KIND_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_KIND_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>               /* the kind is a fixed four bytes, and the node's size rests on it */

/* ══ WHAT A THING IS ══════════════════════════════════════════════════════════════════════════════════
 *
 * Every value in this language is a node, and the first word of a node is its KIND. Everything that
 * looks at one asks this question first: the allocator asks it to know which provider a thing is carved
 * from and how much room it takes, the release asks it to know what to take apart, a copy asks it to
 * know what to follow, and the evaluator asks it to know whether it is looking at something to run or
 * something already run.
 *
 * ⭐ THIS FILE DECLARES THE TYPE AND NOT THE VALUES, AND THE SPLIT IS A PHASE RATHER THAN A PREFERENCE.
 * A package's kinds are rows in its own contract, and the constants are generated from every package's
 * rows at once — which can only happen once every package has written them, and a package is read before
 * some of the others. So the values arrive in a phase of their own, ahead of every header, and this file
 * is read inside the phase after it.
 * ⛳ WHICH MEANS THIS IS THE ONE FILE IN THE PACKAGE THAT CANNOT NAME WHAT IT NEEDS AT ITS OWN TOP. The
 * file that makes the constants reads EVERY package, so including it from inside one would paste a macro
 * the packages listed later have not defined yet — a syntax error inside whichever package happened to be
 * next, pages from the cause. ⛳ THE FAILURE IF THE PHASE IS EVER DROPPED IS HONEST, WHICH IS WHY THIS IS
 * LIVEABLE: the guard at the foot names a constant, so a build that skipped the phase stops HERE, on the
 * line that wanted it, rather than somewhere downstream.
 *
 * ⛳ AND THE PACKAGE OWNS THE WORD `kind` ALREADY, WHICH IS WHY NOTHING HERE IS INVENTED. The parameters
 * are spelled `kind` (`sys__heap__provider`, `sys__heap__object_nodes`), the C interface is
 * `eng_abi_kind_count/name/id`, and what the host binding publishes is `kinds()`. This is the type
 * finally being called what every caller of it already called it.
 * ⛳ THE NAME IS THE LANGUAGE'S AND NOT ONE PACKAGE'S COLLECTION. `sys` IS the language, so `sys__kind`
 * is the type a kind is held in, whoever registered it — `nn`'s kinds are values of this type exactly as
 * the language's own are, and there is no second type for them to be.
 *
 * ── WHY IT IS A FIXED WIDTH AND NOT AN ENUM ─────────────────────────────────────────────────────────
 * A kind carries its package in the top byte and its own number in the three below. One byte of package
 * and three of kind is forced rather than chosen: the field is four bytes, and the node it sits in is a
 * fixed size that the host side indexes by.
 * ⛔ AN ENUM CANNOT HOLD THAT ACROSS PACKAGES. Without a stated underlying type it has only the range its
 * own members need, and a tagged kind falls outside it — `MEASURED` under clang (`hipcc`), where it is a
 * HARD ERROR rather than a warning:
 *     error: integer value 16777219 is outside the valid range of values [0, 31]
 *            for the enumeration type [-Wenum-constexpr-conversion]
 * ⇒ ★★ AND `g++` ACCEPTS THE SAME LINE SILENTLY, WITH EVERY WARNING ASKED FOR, IN A CONSTANT
 * EXPRESSION. Two compilers, opposite answers, and a header both of them read has to satisfy the stricter.
 * Stating the type answered this while one package wrote every row;
 * a constant of a fixed width answers it for everybody, and needs no guard to prove it did.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
typedef uint32_t sys__kind;

/* ⛔ ZERO IS `STANDARD`, AND NOT BECAUSE UNWRITTEN ROOM READS IT — NOTHING CLEARS A NODE'S KIND. A
 * claimed chunk arrives holding whatever its last tenant left, so any kind at all can turn up in room
 * nobody wrote, zero among them. What holds zero here is that `STANDARD` is an executable form with
 * nothing said about it, and the answer to a question about a thing that is NOT THERE is a different
 * sentence that a different number has to carry — which is why `INVALID` sits after the kinds rather
 * than in front of them.
 * ⛳ AND IT IS THE PHASE CHECK AS WELL AS THE VALUE CHECK, for free: it names a generated constant, so a
 * build that never ran the phase that generates them fails on this line. */
static_assert(SYS__KIND__STANDARD == 0u,
              "zero is a kind, and the answer to a question about a thing that is not there is not one: "
              "INVALID belongs after the kinds rather than in front of them, and this holds it there.");

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_KIND_CUH */
