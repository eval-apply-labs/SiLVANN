#ifndef SILVANN__PACKAGES_NN_CPU_PACKAGE__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_PACKAGE__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/package__header.cuh"   /* the hatch this package's own rows are declared into */
#include "../contracts/objects/package.cuh" /* its constants and fault words */
/* ══ nn — STANDING THIS PACKAGE UP ═══════════════════════════════════════════════════════════════════
 *
 * One verb, called once, from the list the boot evaluates after the language is whole. The mechanism and
 * the reasoning behind it belong to the language and are written next door, in `sys/cpu/package__header.cuh`;
 * what is here is this package's own answer to it.
 *
 * ⛳ IT IS DECLARED HERE AND DEFINED BESIDE THIS, for the reason that decides the split everywhere in
 * this tree, and for a second one that is particular to an init: its body names another package's VERB
 * ID, and a verb id is generated from every package's rows at once, in a phase after every header has
 * been read. A definition that names one therefore cannot sit in a header — it would be reaching for a
 * constant that does not exist yet, and the compiler would say so about the wrong file.
 *
 * ── HOW MUCH ROOM THIS PACKAGE ASKS FOR ──────────────────────────────────────────────────────────────
 * The bytes a model's activations are carved from. They are not the heap's: a heap chunk is claimed PER
 * BLOCK, so housing a grid-wide thing there is paid once per block, while a span is paid once — which is
 * the difference between fifteen megabytes and nine hundred and sixty on a real model.
 *
 * ⭐⭐ THERE IS NO SIZE CONSTANT HERE, AND THAT IS THE CONTRACT RATHER THAN AN OMISSION. nn's host init
 * SUMS the four terms its config spells and asks for that total, so this package has no default span for
 * a config to fall back to and no figure in a header that could disagree with one in a file.
 * ⇒ ★ THE ONLY FIGURE THAT CANNOT BE WRONG IS THE ONE NOBODY WRITES.
 * ⛳ A CONFIG NAMING NO nn KEY THEREFORE TAKES NO SPAN AT ALL, which is a state the rest of this package
 * is written to stand in — ▶ the opcode init, which settles rather than refusing when its hatch entry is
 * empty. ▶ `docs_history_internal/nn_retired_symbols.md` for what this replaced and what that cost.
 *
 * ⛳ THE TERM COUNT BELOW IS NOT A SIZE — it is how many figures the span is the sum of, and it lives here
 * because the loop that reads them and the loop that counts them must agree. Adding a fifth term means
 * adding a reader beside it, which is a code change either way; what must never happen is a reader added
 * and the count left behind, so they are written against one name.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
SYS__PACKAGE__DECLARE_HATCH_ROWS(nn)

/* Stand this package up. Takes the init list, so it can ask what has not run yet, and answers nothing —
 * whether it did its work or put itself back at the tail to wait for the language. */
static __device__ __noinline__ void nn__package__zzpackage_apply_init(sys__heap_node* base, uint64_t form);

#endif /* SILVANN__PACKAGES_NN_CPU_PACKAGE__HEADER_CUH */
