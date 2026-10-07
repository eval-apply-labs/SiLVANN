#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH

/* This file needs nothing: a macro body names nothing until something expands it. */
/* ── THE WORDS THIS PACKAGE PUBLISHES INTO THE LANGUAGE ──────────────────────────────────────────────
 * One row per verb a program may call, and the same arrangement as the kinds: the package numbers them
 * from zero in its own file and the machinery adds the package id, so two packages cannot collide and
 * neither writes the other's number.
 *
 *     base        this package's own number for the verb — yours, unique only within the package
 *     name        what the verb is called in the language, QUALIFIED BY THE PACKAGE THAT PUBLISHES IT
 *
 * ⛔⛔ AND THE QUALIFICATION IS THE POINT, BECAUSE THE PACKED ID DOES NOT REACH FAR ENOUGH. Two packages
 * cannot collide in the DISPATCH — their verbs are numbered apart and the switch tells them apart without
 * either knowing. But a program does not write a number, it writes a WORD, and two packages both
 * publishing `clone` would collide in the one place the packing never touches: the spelling a composer
 * has to resolve. ⇒ ★ AN ID SEPARATES WHAT THE MACHINE READS; ONLY A NAME SEPARATES WHAT A PERSON WRITES.
 *
 * ⛳ ONE `__` AND NOT TWO, WHICH IS DELIBERATE. A C name is `package__component__verb`, because a reader
 * of the source wants to know which file to open. A program does not have files; it has words. So the
 * published spelling is `package__verb`, and the component stays where it is useful.
 * ⭐ AND HERE IS WHAT "USEFUL" TURNS OUT TO MEAN, now that there is a case on each side of it. `clone`,
 * `type` and `create` drop theirs because `heap_object` names the FILE and the verb is generic — a program
 * cloning something does not care where the code lives. The register's two keep theirs because
 * `system_register` names the SUBJECT: the verb is not "get", it is "get from the system register", and
 * `sys__get` would be a word too general to belong to one container.
 * ⇒ ★ THE TEST IS WHETHER THE COMPONENT IS A PLACE OR A THING. A place is a hint about the source and
 * goes; a thing is part of what the program is saying and stays.
 *
 * ⚠ WHETHER `sys` KEEPS ITS PREFIX IS OPEN, AND BOTH ANSWERS ARE DEFENSIBLE. ⚖ ARCHITECT: *"maybe we can
 * exempt sys, but for sure ai_silvann should."* FOR the exemption: `sys` IS the language, and every lisp
 * writes `(car x)` rather than `(scheme:car x)` — unqualified means the base and qualified means a
 * library, which is what people already expect. AGAINST: an exemption is a rule with a hole in it, and
 * the hole would be in the package most likely to be read as the example.
 * ⚖⚖ RULED S99 — *"sys__ remains for clarity."* The question is CLOSED, not deferred: every published
 * spelling carries the prefix, `sys`'s own included. ⛳ BOTH ARGUMENTS ARE KEPT ABOVE rather than
 * deleted, because the next package's author will meet the same choice and should meet the reasoning
 * rather than only the outcome — and because the one thing that would reopen it is a person writing
 * programs and finding the prefix in the way, which is evidence nobody has yet.
 * ⛳ The earlier holding position was *"lets keep it consistent for now and then we'll decide"*, and the
 * cost asymmetry that made waiting safe is what makes deciding safe now: unqualifying is one line per
 * row, requalifying after programs exist is not.
 *     apply       the wrapper, which has the one shape every published verb has:
 *                 `void (sys__heap_node* base, uint64_t form)` — the second argument is the form's own
 *                 offset in the heap, and the answer goes back INTO the form through
 *                 `sys__opcodes__becomes` rather than out through a return, so the arm the switch
 *                 generates is a call and not an assignment
 *
 * ⚠ AND THE NAME COLUMN IS PUBLISHED SURFACE, WHICH IS SAID HERE RATHER THAN LEFT TO BE NOTICED. The
 * DISPATCH reads the number and the wrapper and discards the spelling, but the spelling leaves the build
 * anyway: `ENG__ABI__VERB_ROW` in `engine/abi.cuh` gathers these rows into a {spelling, id} table, the C
 * interface publishes it through `eng_abi_verb_count`/`_name`/`_id`, pybind hands it over as `verbs()`,
 * and the composer in `backstage/python/silvann_runtime/composer.py` resolves every word a program writes through that
 * map. ⇒ ★ A WORD TYPED IN A TEST REACHES A VERB BY THIS COLUMN AND BY NOTHING ELSE.
 * ⛳ IT IS NOT THE DANGEROUS KIND OF ROW — that one is a method that runs and does nothing, and leaks while
 * every counter balances. A spelling cannot fail; it can only go stale, and it goes stale the moment
 * somebody renames a verb without touching it — which surfaces as a composer that cannot find the word,
 * far from the row that moved.
 * ⛔ NOTHING CHECKS THE THREE AGREE, AND A GATE FOR IT NEEDS NOTHING THIS TREE LACKS. `REASONED`, premise:
 * every row today derives from its neighbours — `sys__<component>__zzpackage_apply_<verb>` pairs with
 * `SYS__<COMPONENT>__<VERB>`, and the spelling is that pair with the component kept or dropped by the PLACE
 * OR THING test above. A checker over the rows is writable now; what it cannot do is guess which of the
 * two spellings a verb wants, so the row stays the place that says.
 *
 * ⛳ AND WHAT A PROGRAM NEEDS IN ORDER TO USE THESE ROWS IS ALREADY PUBLISHED. `type` answers with a kind
 * and `create` takes one, so both need a program to be able to SAY a kind — and it can: every kind's row
 * in `sys__LANGUAGE_CONTRACT__KINDS` (`language_contract__kinds.cuh`) carries the spelling the host
 * binds, so a composer names the kind and gets the same number this package switches on. Every other
 * package's kinds are gathered beside `sys`'s, over `PACKAGE_LIST`, in `language_contract_kinds.cuh`. */
/* ⛔⛔ ONE ARITHMETIC ROW IS INDIRECTED, AND THE REASON IS AN EXPERIMENT THAT HAS NEVER BEEN VALIDLY
 * MEASURED. `sys__int__add` is the only verb in the tree with a second body: the `SILVANN_ONE_READ`
 * one-read operand fetch (`opcodes.cuh`), armed by `-DSILVANN_ONE_READ=1`. ⚖ *"do try a poc about
 * changing a opcode to c parameter assignment so you do only one read."*
 * ⛳ THE PoC IS LIVE AND UNMEASURED — no valid A/B of it exists — so repointing this row unconditionally
 * would delete an experiment rather than conclude it.
 * ⇒ ⛳ SO THE ROW NAMES THE PoC WHEN THE PoC IS ARMED AND THE ABI ADAPTER OTHERWISE. Both arms are real
 *   builds and `verb_never_abandons_form` follows both.
 * ⚠ AND THE TWO DO NOT COMPETE ON THE SAME AXIS: the PoC removes locates by walking the form ONCE; the
 *   ABI removes them by never giving the verb a form at all.
 * ⛳ `defined()` IS TESTED EXPLICITLY because this file is included BEFORE `opcodes.cuh`, where the
 *   shipping default lives — an `#if SILVANN_ONE_READ` alone would read an undefined name here. It
 *   happens to yield the right answer; writing it out means it does not depend on happening to. */
#  define SYS__OPCODES__ZZABI_INT_ADD_APPLY  sys__opcodes__zzabi_adapter_int_add

#define sys__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                                \
    X(PKG, 0, "sys__clone",  sys__opcodes__zzabi_adapter_clone,        SYS__HEAP_OBJECT__CLONE)                     \
    X(PKG, 1, "sys__type",   sys__opcodes__zzabi_adapter_type,         SYS__HEAP_OBJECT__TYPE)                      \
    X(PKG, 2, "sys__create", sys__opcodes__zzabi_adapter_create,       SYS__HEAP_OBJECT__CREATE)                    \
    /* Reading and writing a row of the system register. They are published for the reason the register  */ \
    /* exists at all: what a scope cannot own, a program still has to be able to reach — and the only    */ \
    /* way in is these, from the language exactly as from C.                                            */ \
    X(PKG, 3, "sys__system_register__get", sys__opcodes__zzabi_adapter_register_get,                        \
      SYS__SYSTEM_REGISTER__GET)                                                                       \
    X(PKG, 4, "sys__system_register__set", sys__opcodes__zzabi_adapter_register_set,                        \
      SYS__SYSTEM_REGISTER__SET)                                                                             \
    /* Assignment. It is published where reading a name is not, because a program ASKS to assign and     */ \
    /* only MENTIONS a name — so the read is the evaluator's own step and has no caller to be given one. */ \
    X(PKG, 5, "sys__bindings__set", sys__opcodes__zzabi_adapter_bindings_set,                                      \
      SYS__BINDINGS__SET)                                                                                    \
    /* What a first program needs. `let` binds inline and leaves a `remove_bindings` behind it, and both   */ \
    /* of those are published on their own too, so a scope is an instruction in the list rather than a    */ \
    /* shape in the evaluator's control flow — whether a program writes it or a `let` does.               */ \
    /* ⛳ 11 IS NOT REUSED: a number a program has held must not come to mean a different verb.            */ \
    X(PKG, 6, "sys__add_bindings",    sys__opcodes__zzabi_adapter_add_bindings,      SYS__OPCODES__ADD_BINDINGS)    \
    X(PKG, 7, "sys__remove_bindings", sys__opcodes__zzpackage_apply_remove_bindings, SYS__OPCODES__REMOVE_BINDINGS) \
    X(PKG, 8, "sys__eq",              sys__opcodes__zzabi_adapter_eq  ,              SYS__OPCODES__EQ)              \
    X(PKG, 9, "sys__add",             sys__opcodes__zzabi_adapter_add,               SYS__OPCODES__ADD)             \
    X(PKG, 10, "sys__if",             sys__opcodes__zzabi_adapter_if,                SYS__OPCODES__IF)              \
    X(PKG, 12, "sys__begin",          sys__opcodes__zzpackage_apply_begin,           SYS__OPCODES__BEGIN)         \
    /* What a CALL is made of. `prog1` is `begin`'s mirror and exists because a call's value has to      */ \
    /* survive the unbinding that follows it; `defun` binds a name to a procedure, because a defun's     */ \
    /* destination is lexical and there is no table of functions anywhere.                              */ \
    X(PKG, 13, "sys__prog1",          sys__opcodes__zzpackage_apply_prog1,           SYS__OPCODES__PROG1)         \
    X(PKG, 14, "sys__sub",            sys__opcodes__zzabi_adapter_sub,               SYS__OPCODES__SUB)           \
    X(PKG, 15, "sys__less",           sys__opcodes__zzabi_adapter_less,              SYS__OPCODES__LESS)          \
    /* ⭐⭐⭐ THE TYPED ARITHMETIC — ⚖ RULED: *"let's simply publish type specific additions, */ \
    /* so sys__int__add and sys__float__add, and add will take the first element, check its type and   */ \
    /* run the add for that type … you are guaranteed to have as an output the same data type as the   */ \
    /* first element."*                                                                                */ \
    /* ⭐⭐ SO THE ASYMMETRY IS THE CONTRACT AND NOT A GAP: *"a float adding an int still works, but an */ \
    /* int adding a float won't."* A float WIDENS its partner, which is lossless; an int would have to */ \
    /* NARROW one, which is not. ⇒ ★ THE DIRECTION THAT KEEPS EVERY BIT IS THE DIRECTION THAT IS       */ \
    /* ALLOWED, and the first operand choosing the domain is what makes that decidable by reading.     */ \
    /* ⛳ AND `sys__add` IS THE WORD A PROGRAM WRITES — it dispatches on the first operand's kind, it  */ \
    /* does not compute, so an int program reduces through `sys__int__add` whatever it was frozen as. */ \
    X(PKG, 25, "sys__int__add",       SYS__OPCODES__ZZABI_INT_ADD_APPLY,             SYS__OPCODES__INT_ADD)       \
    X(PKG, 26, "sys__float__add",     sys__opcodes__zzabi_adapter_float_add,         SYS__OPCODES__FLOAT_ADD)     \
    X(PKG, 27, "sys__int__sub",       sys__opcodes__zzabi_adapter_int_sub,           SYS__OPCODES__INT_SUB)       \
    X(PKG, 28, "sys__float__sub",     sys__opcodes__zzabi_adapter_float_sub,         SYS__OPCODES__FLOAT_SUB)     \
    X(PKG, 29, "sys__int__less",      sys__opcodes__zzabi_adapter_int_less,          SYS__OPCODES__INT_LESS)      \
    X(PKG, 30, "sys__float__less",    sys__opcodes__zzabi_adapter_float_less,        SYS__OPCODES__FLOAT_LESS)    \
    X(PKG, 16, "sys__defun",          sys__opcodes__zzabi_adapter_defun,             SYS__OPCODES__DEFUN)         \
    /* A scope, as one word rather than three forms a program has to place in the right order. It is the   */ \
    /* environment's, beside `set!`, because binding and unbinding is what it does.                        */ \
    X(PKG, 17, "sys__let",            sys__bindings__zzpackage_apply_let,            SYS__BINDINGS__LET)          \
    /* Dispatch, and the one word that makes an argument for it. A program sends a COPY of its program   */ \
    /* and a COPY of what its names mean; the block that receives them builds its own of each, so two    */ \
    /* blocks share a snapshot and neither writes anything the other reads.                              */ \
    X(PKG, 18, "sys__bindings__viewonly", sys__opcodes__zzabi_adapter_viewonly,  SYS__BINDINGS__VIEWONLY)  \
    X(PKG, 19, "sys__compute",        sys__opcodes__zzabi_adapter_compute,  SYS__COMPUTING_BASE__COMPUTE) \
    X(PKG, 20, "sys__result",         sys__opcodes__zzabi_adapter_result,   SYS__COMPUTING_BASE__RESULT) \
    X(PKG, 21, "sys__completed",      sys__opcodes__zzabi_adapter_completed, SYS__COMPUTING_BASE__COMPLETED) \
    /* Standing this package up, which is a call in a list the boot evaluates rather than a C entry     */ \
    /* point the boot knows the name of. It is published like any other word because that is how a      */ \
    /* package hands one over, and because a package that wants to run another's init has no other      */ \
    /* way to name it — which is the same reason the list can be asked what has not run yet.            */ \
    X(PKG, 22, "sys__package__init",  sys__opcodes__zzabi_adapter_package_init,            SYS__PACKAGE__INIT) \
    /* ⭐⭐⭐ THE DICTIONARY, REACHABLE FROM A PROGRAM — ⚖ *"publish the verbs"*. The settings reader,  */ \
    /* the symbol tables and the bindings all run on it in C, and these rows are what let a program ask */ \
    /* it anything. ⇒ ★ A STRUCTURE EVERY INTERNAL CALLER USES AND NO PROGRAM CAN REACH IS A LIBRARY,   */ \
    /* NOT A LANGUAGE FEATURE.                                                                          */ \
    /* ⚖ WHAT IT WAS PUBLISHED FOR: the dispatch table `nn` publishes and an extension package REWORKS  */ \
    /* — *"by modifying that list it can do code injection"* — keyed on the expert-type enum, *"a       */ \
    /* dictionary so key val is less punitive"*. ⚖ That table is v2; the two verbs stand   */ \
    /* on their own. ⛳ TWO ROWS AND NOT THREE: `sys__create` is generic and the dictionary's OBJECTS    */ \
    /* row already names its constructor, so `(create <dictionary>)` needs no row of its own.           */ \
    X(PKG, 23, "sys__dictionary__put", sys__opcodes__zzabi_adapter_dict_put,          SYS__DICTIONARY__PUT) \
    X(PKG, 24, "sys__dictionary__get", sys__opcodes__zzabi_adapter_dict_get,          SYS__DICTIONARY__GET) \
    /* ⭐ AN ELEMENT OF A NODE ARRAY, BY POSITION — what a program reads a card's answer out of when the  */ \
    /* card wrote it into nodes the program made (⚖ *"allocate the return nodes beforehand and send the */ \
    /* node array base address"*): `nn__vector__top_k`'s picks, expert `j` of them.                     */ \
    X(PKG, 31, "sys__node_array__get", sys__opcodes__zzabi_adapter_array_get,         SYS__NODE_ARRAY__GET) \
    /* ⭐ AND ITS MIRROR, THE WRITE — what a boot program fills a table with: ⚖ *"plane table"*, a layer's  */ \
    /* weights as a node array one verb is handed. The array owns what it holds, so storing takes a hold. */ \
    X(PKG, 32, "sys__node_array__set", sys__opcodes__zzabi_adapter_array_set,         SYS__NODE_ARRAY__SET) \
    /* ⭐ A FILE, READ-ONLY, AS A HANDLE — ⚖ *"sys door"*: what a boot program names a model's weights on disk */ \
    /* with, so a verb can read an expert that is not in memory. The reading itself is the family's door.  */ \
    X(PKG, 33, "sys__file__open",     sys__opcodes__zzabi_adapter_file_open,         SYS__FILE__OPEN)        \
    X(PKG, 34, "sys__file__close",    sys__opcodes__zzabi_adapter_file_close,        SYS__FILE__CLOSE) \
    /* ⭐ WHICH WORKER THIS IS — ⚖ *"every worker has an id that corresponds to its array index in the         */ \
    /* computing_base"*: a device of a family, and block `w` runs as worker `w`.                             */ \
    X(PKG, 35, "sys__worker",         sys__opcodes__zzabi_adapter_worker,            SYS__WORKER) \
    /* ⭐ A QUOTED FORM RUN WHERE IT STANDS — what `if` does to the arm it picks, on its own.               */ \
    X(PKG, 36, "sys__unquote",        sys__opcodes__zzabi_adapter_unquote,           SYS__UNQUOTE) \
    /* ⭐ A LOOP IN ONE FORM — the verb rewrites its own form each turn, so a loop of any length holds what  */ \
    /* one turn holds. As a defun each turn nested inside the last and the heap ran out in ~1600 turns.   */ \
    X(PKG, 37, "sys__while",          sys__opcodes__zzpackage_apply_while,           SYS__WHILE)

/* ── AND THE OPCODES, NAMED ──────────────────────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"the enum addition on the opcodes."* A published verb had a number and no name, so every
 * caller — a composer, a test, a person reading a dispatch — had to carry the mapping itself. These are
 * the names, generated from the same rows the dispatch is, so a verb cannot be named one thing here and
 * dispatched as another.
 *
 * ⭐ THE VALUE IS THE PACKED ID AND NOT THE ROW'S BASE, because the packed id is what a program holds and
 * what the switch compares. A row carries a base; the package is applied HERE, once, at the gather —
 * which is the arrangement the tree already uses everywhere and the one that stopped 33 opcodes escaping
 * untagged when it was applied per-row instead.
 * ⛳ AND THEY ARE `constexpr uint64_t` RATHER THAN AN ENUM, because a second package's verbs carry its id
 * at bit 32 and would not fit the type a plain enum picks for small values.
 *
 * ⛳ EACH IS NAMED FOR THE COMPONENT THAT DEFINES IT AND FOR THE VERB IT IS — `SYS__<component>__<verb>`,
 * with nothing between them. Every other constant here is owned that way, from a fault word to a
 * geometry, so a reader who has the name knows which file to open; a flat prefix would name a space that
 * does not exist, since the ids are packed per package rather than dealt from one pool.
 * ⛔ AND NOTHING SAYS "VERB" IN IT, WHICH IS WORTH ONE LINE BECAUSE I PUT IT THERE. These are verbs
 * because they are named for what they DO — clone, type, create, get, set — and a word repeating that
 * classification carries no information a reader did not already have from reading the name.
 *
 * ⛳ THE NAMES THEMSELVES ARE GENERATED ONE FLOOR UP, in `language_contract_verbs.cuh` beside the roster,
 * from the rows every package writes. They cannot be generated here: a gather that reads EVERY package's
 * rows has to run after every package has written them, and this file is one package's.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
