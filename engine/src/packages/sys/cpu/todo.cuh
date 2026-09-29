#ifndef SILVANN__PACKAGES_SYS_CPU_TODO_CUH
#define SILVANN__PACKAGES_SYS_CPU_TODO_CUH

/* This file needs nothing. Every entry in it is a signature and a stand-in, and a stand-in has no
 * dependencies by construction: a stand-in that needs something is not a stand-in. */

/* ══ WHAT IS NOT BUILT YET, DECLARED SO THAT CODE CAN BE WRITTEN AGAINST IT ═══════════════════════════
 *
 * ⚖ ARCHITECT: *"mark it into a todo.cuh so you can import it as a signature, this also means we can grep
 * who imports it and make function mockups that can be replaced later."*
 *
 * ⭐⭐ INCLUDING THIS FILE IS A DECLARATION OF DEBT, AND THAT IS THE MECHANISM. A caller that needs an
 * answer nobody can give yet does not invent one, guess one, or leave the question in a comment where
 * nothing can find it. It includes this file, and `grep -l todo.cuh` is then the complete list of code
 * standing on something unbuilt. When the real thing lands, the entry leaves and every dependent stops
 * compiling until it is pointed at the real one — which is the loudest way for a debt to be repaid.
 *
 * ── THE THREE RULES ─────────────────────────────────────────────────────────────────────────────────
 * ⛔ A STAND-IN MUST NOT BE MISTAKEABLE FOR AN ANSWER. It returns a value that means "I do not know" and
 *    that a caller has to handle, never a plausible number. A stand-in that returns something reasonable
 *    is worse than no stand-in: it makes every caller correct-looking and every result wrong.
 * ⛔ EVERY ENTRY NAMES WHO REPLACES IT. Not "later" — the thing that has to exist, so that whoever builds
 *    that thing finds this by grepping for it rather than by remembering.
 * ⛔ AN ENTRY LEAVES WHEN ITS SUBJECT ARRIVES. This file shrinks; it is not a backlog and it is not a
 *    place to record intentions. If it cannot be written as a signature, it does not belong here.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛳ NOTHING IS OWED RIGHT NOW. The file stays because the convention is worth having ready — an empty
 * registry costs a line in the manifest and the next debt gets a home instead of a comment.
 *
 * ⛳ ITS FIRST TENANT LASTED ONE SESSION AND THAT IS THE MECHANISM WORKING, not a sign it was
 * unnecessary. `sys__todo__grid_width` stood in for a number nobody could answer; writing the caller
 * against a signature is what made it obvious that the question was a SILICON one, and the seam already
 * had a place for it. ⇒ ★ A DEBT WITH A SIGNATURE GETS LOOKED AT; A DEBT IN A COMMENT GETS READ PAST. */

#endif /* SILVANN__PACKAGES_SYS_CPU_TODO_CUH */
