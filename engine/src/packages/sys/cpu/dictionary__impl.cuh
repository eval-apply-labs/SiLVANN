#ifndef SILVANN__PACKAGES_SYS_CPU_DICTIONARY__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_DICTIONARY__IMPL_CUH
/* ══ sys — WHY THE DICTIONARY'S PUBLISHED VERBS ARE NOT BESIDE IT ═══════════════════════════════════
 *
 * ⛔ THIS FILE DEFINES NOTHING. The two verbs' bodies are `sys__opcodes__zzabi_dict_put` / `_get`, in
 * `opcodes/opcodes_abi.cuh`, behind the bridge every ruled-shape verb uses.
 * ⛳ WHY THEY ARE NOT IN `dictionary.cuh`, WHEN THE DICTIONARY IS OTHERWISE ONE FILE. `dictionary.cuh` is a
 * HEADER-PHASE file — the settings reader, the symbol tables and the bindings all need it standing
 * before they are declared — and it therefore sits BELOW `sys__sublist__*` and `sys__opcodes__*` in the
 * include order. An apply verb needs both. ⇒ ★ THE PHASE A FILE IS IN IS A FACT ABOUT WHO NEEDS IT
 * FIRST, NOT ABOUT WHAT IT CONTAINS, so a structure can be header-phase and its language surface cannot.
 * ⛔ FOUND BY THE COMPILER AND NOT BY READING: putting these beside the structure fails with
 * *"use of undeclared identifier 'sys__sublist__length'"*, which is the include order saying so.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh" /* the hold a `get` hands over, and the release that gives it back */
#include "dictionary.cuh"          /* the structure these two verbs are the language surface of */
#include "list__header.cuh"        /* what a published verb is handed */
#include "opcodes/opcodes.cuh"             /* the two lines every verb ends with, and the truth objects */
/* ══ ⭐⭐⭐ WHAT THE LANGUAGE CALLS — ⚖ *"publish the verbs"* ═════════════════════════════
 *
 * ⛔⛔ THE STRUCTURE'S C SURFACE IS WIDE AND ITS LANGUAGE SURFACE IS TWO WORDS. `create`, `put`, `get`,
 * `remove`, `count`, `seal` and `number_key` all ship, and the settings reader, the symbol tables and the
 * bindings machinery all run on them; a PROGRAM reaches `put` and `get`. ⇒ ★ A STRUCTURE EVERY INTERNAL
 * CALLER USES AND NO PROGRAM CAN REACH IS A LIBRARY, NOT A LANGUAGE FEATURE, and the gap is invisible
 * from inside because every internal caller is satisfied.
 *
 * ⚖ WHAT IT IS FOR: *"nn will publish a let list of key/val that contains the enum and the function that
 * handles it so nn_tq by modifying that list can do code injection"*, and ⚖ *"the dispatch table should
 * be a dictionary so key val is less punitive"*. A package extends another package's behaviour by
 * EDITING A TABLE IN THE LANGUAGE rather than by touching its C — and that needs exactly these two doors.
 *
 * ⛳ ONLY TWO, BECAUSE THE THIRD IS GENERIC: `sys__create` works on any kind and the dictionary's OBJECTS
 * row names `sys__dictionary__zzpackage_construct` as its constructor, so `(create <dictionary>)` needs
 * nothing of its own. ⇒ ★ COUNTING THE DOORS A FEATURE NEEDS MEANS CHECKING WHICH ARE ALREADY OPEN; the
 * obvious answer here was three.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#endif /* SILVANN__PACKAGES_SYS_CPU_DICTIONARY__IMPL_CUH */
