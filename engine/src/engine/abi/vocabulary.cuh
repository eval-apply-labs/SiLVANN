#ifndef SILVANN__ENGINE_ABI_VOCABULARY_CUH
#define SILVANN__ENGINE_ABI_VOCABULARY_CUH

/* ══ THE VOCABULARY — VERBS, KINDS AND PACKAGES, ENUMERABLE ═══════════════════════════════════════════
 * The tables generated from the lists that define the language's words, kinds and packages, and the
 * doors that enumerate them. */

/* ── THE VOCABULARY, GENERATED FROM THE LISTS THAT DEFINE IT ─────────────────────────────────────────
 * ⭐⭐ WITHOUT THIS THE INTERFACE ABOVE CANNOT BE USED TO WRITE A PROGRAM, WHICH IS EASY TO MISS. A
 *   caller can make a list and put cells in it, and a cell needs a KIND and sometimes a VERB — both of
 *   which are numbers it has no way to learn. Publishing the operations and withholding the words is an
 *   interface that compiles and cannot be called.
 * ⭐ AND NOT ONE NUMBER IS TYPED HERE. Both tables are the package's own lists expanded with an arm that
 *   takes a row and contributes an entry, so a verb added to a package arrives here for free and one
 *   spelled differently cannot disagree with itself. A hand-written table would be a second copy of a
 *   fact that already has an owner. ⇒ ★ A LIST CANNOT FORGET; A SWEEP CAN.
 * ⛳ BOTH TABLES ARE GATHERED PER PACKAGE, through the same roster everything else is gathered through,
 *   so a second package's words and its kinds appear without this file learning its name.
 * ⛔ THE KIND TABLE AND THE VERB TABLE ARE ONE GATHER OVER ONE ROSTER. A table that read one package's
 *   list would let a kind registered anywhere but the language dispatch correctly and arrive here
 *   unnamed — a table that names one package is a table that has to be edited when another arrives. */
typedef struct { const char*        name; unsigned long long id; } EngineAbiVerb;
typedef struct { const char*        name; unsigned int       id; } EngineAbiKind;

#define ENG__ABI__VERB_ROW(PKG, BASE, SPELLING, APPLY, NAME, ...)  { SPELLING, (unsigned long long)NAME },
#define ENG__ABI__VERB_GATHER(ID, NAME, ...)  NAME##__LANGUAGE_CONTRACT__VERBS(ENG__ABI__VERB_ROW, NAME)
static const EngineAbiVerb eng__abi__zzprivate_verbs[] = { PACKAGE_LIST(ENG__ABI__VERB_GATHER) };
#undef ENG__ABI__VERB_GATHER
#undef ENG__ABI__VERB_ROW

#define ENG__ABI__KIND_ROW(PKG, BASE, SPELLING, NAME)  { SPELLING, (unsigned int)NAME },
#define ENG__ABI__KIND_GATHER(ID, NAME, ...)  NAME##__LANGUAGE_CONTRACT__KINDS(ENG__ABI__KIND_ROW, NAME)
static const EngineAbiKind eng__abi__zzprivate_kinds[] = { PACKAGE_LIST(ENG__ABI__KIND_GATHER) };
#undef ENG__ABI__KIND_GATHER
#undef ENG__ABI__KIND_ROW

/* ⛳ AND THE ROSTER ITSELF, WHICH IS THE ONE TABLE HERE THAT NEEDS NO GATHER — a package's id and name
 * are columns of `PACKAGE_LIST` already, so this reads the roster directly rather than asking every
 * package for rows. ▶ `eng_abi_package_count` below for what asked for it. */
typedef struct { const char*        name; unsigned int       id; } EngineAbiPackage;
#define ENG__ABI__PACKAGE_ROW(ID, NAME, ...)  { #NAME, (unsigned int)(ID) },
static const EngineAbiPackage eng__abi__zzprivate_packages[] = { PACKAGE_LIST(ENG__ABI__PACKAGE_ROW) };
#undef ENG__ABI__PACKAGE_ROW

extern "C" {

/* ── AND THE WORDS, ENUMERABLE RATHER THAN LOOKED UP ─────────────────────────────────────────────────
 * A caller wants all of them once, to build whatever map its own language uses, so counting and indexing
 * is the whole surface. A name-to-id lookup would make the caller ask one question at a time for an
 * answer that never changes, and would put a string comparison on a path that has no need of one. */
unsigned int       eng_abi_verb_count(void) {
    return (unsigned int)(sizeof eng__abi__zzprivate_verbs / sizeof eng__abi__zzprivate_verbs[0]);
}
const char*        eng_abi_verb_name(unsigned int at) {
    return (at < eng_abi_verb_count()) ? eng__abi__zzprivate_verbs[at].name : 0;
}
unsigned long long eng_abi_verb_id(unsigned int at) {
    return (at < eng_abi_verb_count()) ? eng__abi__zzprivate_verbs[at].id : 0ull;
}

unsigned int       eng_abi_kind_count(void) {
    return (unsigned int)(sizeof eng__abi__zzprivate_kinds / sizeof eng__abi__zzprivate_kinds[0]);
}
const char*        eng_abi_kind_name(unsigned int at) {
    return (at < eng_abi_kind_count()) ? eng__abi__zzprivate_kinds[at].name : 0;
}
unsigned int       eng_abi_kind_id(unsigned int at) {
    return (at < eng_abi_kind_count()) ? eng__abi__zzprivate_kinds[at].id : 0u;
}

/* ── AND THE PACKAGES, FOR THE SAME REASON THE WORDS AND THE KINDS ARE HERE ──────────────────────────
 * ⚖ ARCHITECT: *"make the abi publish the roster, after all by the time we are in the abi
 * the setup is built and there is no ambiguity."*
 *
 * ⭐ WHAT ASKED FOR IT IS THE BOOT SPLIT. A package's room is ONE ALLOCATION PER PACKAGE, taken by that
 * package's host init before the machine is booted, and its starting address is published in that package's
 * own extension-hatch entry, which is indexed BY PACKAGE ID, while a config file spells a NAME.
 * ⛔⛔ AND THE FIRST REASON WRITTEN HERE WAS ALREADY WEAKER THAN IT LOOKED, CORRECTED THE SAME DAY. It
 * said this table is *"the only place that knows both"* — which is true of PYTHON and false of the
 * engine, and the engine is where the mapping happens. ⚖ RULED hours later: the ABI allocator loops the
 * roster and calls each package's HOST-SIDE init, so `PACKAGE_LIST` is walked directly in C and the name
 * never has to become an id across a language boundary at all.
 * ⇒ ⭐ WHAT THIS TABLE IS ACTUALLY FOR, WHICH IS A SMALLER AND HONEST CLAIM: it is the ROSTER MADE
 * INSPECTABLE — a caller, a test or a config checker can ask which packages a build has and what they are
 * numbered, the same way it can already ask for the verbs and the kinds. The device suite uses it for the
 * one check nothing else could make: that a kind's top byte is the id the roster publishes
 * (`K["nn__buffer"] >> 24 == P["nn"]`), which is two tables agreeing rather than one asserting.
 * ⇒ ★ A RATIONALE WRITTEN WHILE A DESIGN IS STILL MOVING OUTLIVES THE DESIGN, so it is corrected here
 * rather than left for a reader to take at face value — the surface is worth keeping, the reason was not
 * the one given.
 * ⛳ THE NAME IS THE PASTED TOKEN, `#NAME`, AND NOT THE ROSTER'S THIRD COLUMN — which happens to be the
 * same text today and is not the same FACT. `NAME` is what forms `nn_pkg_id` and `nn__LANGUAGE_CONTRACT__*`,
 * so publishing it is publishing the spelling a package is actually identified by; the third column is a
 * directory, and a directory is free to be renamed without anything else moving. ⇒ ★ PUBLISH THE NAME THAT
 * CANNOT DRIFT FROM THE THING IT NAMES: this one cannot, because the same token builds both.
 * ⛳ COUNTING AND INDEXING IS THE WHOLE SURFACE, exactly as above — a caller wants all of them once, to
 * build the map its own language uses, and a name-to-id lookup here would put a string comparison on the
 * boot path for an answer that never changes. */
unsigned int       eng_abi_package_count(void) {
    return (unsigned int)(sizeof eng__abi__zzprivate_packages / sizeof eng__abi__zzprivate_packages[0]);
}
const char*        eng_abi_package_name(unsigned int at) {
    return (at < eng_abi_package_count()) ? eng__abi__zzprivate_packages[at].name : 0;
}
unsigned int       eng_abi_package_id(unsigned int at) {
    return (at < eng_abi_package_count()) ? eng__abi__zzprivate_packages[at].id : 0u;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_VOCABULARY_CUH */
