#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH

/* THIS FILE NEEDS NOTHING, and that is a property rather than an omission: a macro body names nothing
 * until something expands it, so the methods named below are resolved where the switches are, in the
 * files that expand them — which is what lets this list sit early, where the allocator can read a column
 * out of it.
 * ⛳ AND IT EXPANDS NOTHING ITSELF, WHICH IS WHAT MAKES THE SENTENCE ABOVE TRUE OF THE FILE AND NOT ONLY
 * OF THE ROWS. Anything that runs `PACKAGE_LIST` reads EVERY package's rows, so it has to run after every
 * package has written them — which no package can do, because each is read before some of the others.
 * The expansion that reads them all is `language_contract_verbs.cuh`, beside the roster, which is where
 * anything belonging to every package rather than to one of them lives. */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 *   · The provider was once an argument to `make`, and `make(SYS__KIND__STACK, ARRAYS)` really did put a
 *     stack in the churn chunk; two call sites held the design up by being written correctly.
 *   · The procedure row's note said `defun` never walked the parameters. It has walked them since
 * (`sys__opcodes__zzabi_defun` in `cpu/opcodes/opcodes_abi.cuh`).
 * ⛳ RETIREMENT: at release, with the rest.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ WHAT THIS PACKAGE CONTRIBUTES TO THE LANGUAGE ═══════════════════════════════════════════════════
 *
 * ⭐ WHY THE NAME IS `language` AND NOT `package`. The manifest is already the package↔machinery
 * contract — it says who is listed, in what order, and how the gathers reach them. This file is a
 * different contract with a different counterparty: what a package hands to the LANGUAGE. Kinds a
 * program can hold, words it can call, and types it can be handed. Calling it a package contract would
 * name the mechanism that carries it rather than the thing it carries, and would say `package` twice,
 * since the package's own name is already the first token of every macro below.
 * ⛳ AND IT IS NOT `heap` OR `engine` EITHER, THOUGH EACH IS TRUE OF ONE HALF: the OBJECTS rows are
 * gathered five times and every one of them is the heap's, the VERBS rows three times and every one
 * is the engine's. Two counterparties, one act — so the name is the act's.
 *
 *
 * One file, one macro, one row per kind. It is the package's ANSWER to the object interface declared in
 * `heap_object__header.cuh` — the implements clause — and it is here rather than beside either of them for
 * a reason that is about packages and not about this one.
 *
 * ⭐⭐ THE NAME IS THE MECHANISM. The dispatches gather by pasting a package's name onto
 * `__LANGUAGE_CONTRACT__OBJECTS`, so every package answers at `<name>__LANGUAGE_CONTRACT__OBJECTS` and the switch that
 * expands them all is written once, over `PACKAGE_LIST`, and never edited again as packages arrive.
 * ⚖ ARCHITECT: *"ideally it should be the same name for everyone so the manifest can scout all of the
 * files and place them there."* That is the design and it is already what the gather assumes; what was
 * missing was a HOME for the list, not the convention.
 *
 * ⛔⛔ AND IT MUST NOT GO IN `packages/manifest.cuh`, WHICH WAS THE TEMPTING OPTION AND IS RULED OUT
 * ALREADY. The project guide says it of the dtype enum and it is the same argument here: putting every
 * package's kinds in one shared file means `nn` adds a kind BY EDITING `sys` — the exact
 * cross-package coupling packed opcode ids were built to make impossible. A list per package, gathered at
 * the end, is the same shape as `PACKAGE_LIST` and needs nothing new.
 *
 * ⛳ THE REGISTRY IS BUILT, AND WHAT IS OWED IS SMALLER THAN IT. Every gather in the tree expands
 * `PACKAGE_LIST` over a `..._GATHER_ROW` — the OBJECTS arms in `heap_object__impl.cuh` and `heap__impl.cuh`,
 * the VERBS arms in `heap_object__impl.cuh` and `engine/abi.cuh`, and the TYPES arms in
 * `language_contract_kinds.cuh` and `engine/abi.cuh` — so a second package arrives by being LISTED and no
 * switch here is edited for it.
 * ⛳ AND A PACKAGE WITH NOTHING TO SAY IN A CATEGORY WRITES AN EMPTY MACRO FOR IT RATHER THAN INHERITING
 * ONE. There is no generic default and there cannot be: the macro's name is PASTED from the package's,
 * so there is nothing to guard against until the row exists. An empty definition is therefore its own
 * presence check — a package that reserves an id and publishes nothing fails to compile AT ITS OWN ROW,
 * naming itself, rather than somewhere further down.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE ONE ROW PER KIND ────────────────────────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"the polymorphic method is specified as a tag in the x macro ... because we cannot rely
 * on polymorphism as the names are different because they hold the object inside, so the explicit method
 * in the x macro wins."*
 *
 * ⭐ THE ROW IS DATA AND THE DISPATCHES ARE CODE, WHICH IS WHY THEY ARE IN DIFFERENT FILES. A macro
 * definition names nothing — the methods in it are text until something expands it — so the list can sit
 * HERE, early, where `make` needs the provider column, while the switches that CALL those methods stay in
 * `heap_object__impl.cuh`, after every method exists.
 *
 * ⛳ SIX COLUMNS AFTER THE PACKAGE, AND EACH IS A FACT ABOUT A KIND STATED IN ONE PLACE AND READ FROM IT:
 *
 *     dtype       what the object says it is — YOUR OWN NUMBER, and the machinery makes it unique
 *     release     what it hands over when its last reference goes
 *     provider    which chunk it is carved from — ⚖ *"a scaffolding provider for objects, that is not
 *                 called directly"*: a kind a caller names is an OBJECT, a kind that is another object's
 *                 internal storage is SCAFFOLDING, and a caller does not get to say which
 *     clone       how a copy of it is made
 *     construct   how one is built when a program asks `create` for it, taking the base and the caller's
 *                 parameters — or a NAMED REFUSAL, which is the point of listing a kind that must not be
 *                 built: an absent row falls to a default that cannot tell a decision from an oversight
 *     nodes       how much room one takes, in NODES — the kind's own size and nobody else's
 *                 business. ⛔ ZERO MEANS IT IS NEVER MADE HERE: a chunk is a kind and it is not
 *                 carved, it IS the room, so `make` has nothing to place for it.
 *
 * ⛔ THE PROVIDER IS A COLUMN AND NOT AN ARGUMENT TO `make`, AND THAT CLOSES A HOLE RATHER THAN TIDYING
 * ONE. An argument is something nothing checks: `make(SYS__KIND__STACK, ARRAYS)` would compile, run, and
 * put a long-lived object in the churn chunk — exactly the stranding two providers exist to prevent,
 * silently, with every count balancing. A column is stated once, beside the kind.
 *
 * ⛳ A ROW NAMING A METHOD THAT DOES NOT EXIST WILL NOT COMPILE, because the generated switch CALLS it.
 * That is stronger than the package registry's `_PRESENT` assert, which had to be added by hand. What is
 * NOT caught is the other direction: a new kind with no row falls to a refusal at run time, on whatever
 * path first releases or clones one.
 *
 * ⛔ The rows must NOT go in the shared dtype enum, tempting as that is. Putting them there would mean
 * `nn` adding a kind by editing `sys` — the exact cross-package coupling that packed opcode ids
 * were built to make impossible. A list per package, gathered at the end, is the same shape as
 * `PACKAGE_LIST` and needs nothing new.
 *
 * ── WHAT THE NUMBER IN THE FIRST COLUMN IS, WHICH IS LESS THAN IT LOOKS ─────────────────────────────
 * A kind's number is the package's own, chosen in the package's own file, and it only has to be unique
 * WITHIN the package. What makes it unique everywhere is the package's id, which sits in the top byte
 * and is pasted on by the machinery — so a row never carries it, and there is nothing here anybody could
 * forget to write. The room left for a package's own numbering is three bytes.
 * ⛳ WHICH IS WHY THE ROWS BELOW LOOK LIKE ORDINARY SHARED NAMES AND ARE NOT AN EXCEPTION TO THAT. `sys`
 * is package zero, so its kinds pack to the numbers they already have; the tag is there and it is
 * nothing, which is what let the rule arrive without a single value moving.
 * ⛔ A NUMBER THAT IS RETIRED STAYS RETIRED, and this is the one clause a row-writer can break on their
 * own. Gaps cost nothing — no array is indexed by a kind and no test compares a range of them — but a
 * kind is stamped into a node, and nodes are saved and logged, so a number handed to something new is
 * read as the old thing by every record that still carries it. Leave the gap; it is free.
 * ▶ The whole numbering contract, including what a second package owes, is in `packages/manifest__header.cuh`.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

#define sys__LANGUAGE_CONTRACT__OBJECTS(X, PKG)                                                                   \
    X(PKG, SYS__KIND__STACK,            sys__stack__zzpackage_release_internal,                             \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__stack__zzpackage_construct,   2u)                                                            \
    X(PKG, SYS__KIND__STACK_CHUNK,      sys__stack_chunk__zzpackage_release_internal,                       \
      SYS__COMPUTING_BASE__ARRAYS,       sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   SYS__ALLOC__ARRAY_GRANULE)                                           \
    X(PKG, SYS__KIND__HEAP_OBJECT,      sys__heap_object__release__default,                                  \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__heap_object__create__plain,    2u)                                                            \
    X(PKG, SYS__KIND__COMPUTING_BASE,   sys__computing_base__zzpackage_never_release_internal,              \
      SYS__COMPUTING_BASE__NO_PROVIDER,  sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   2u)                                                            \
    X(PKG, SYS__KIND__ERROR,            sys__heap_object__release__default,                                  \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__error__zzpackage_construct,   2u)                                                            \
    /* ⛳ TEXT HOLDS NOTHING, SO ITS RELEASE IS THE DEFAULT, and a program cannot build one: strings are */ \
    /*   made when a program is COMPOSED, through the C door. Its size is a floor, like the array's — an */ \
    /*   empty string is a head and a hash, and a caller with text asks for more.                        */ \
    X(PKG, SYS__KIND__STRING,           sys__heap_object__release__default,                                  \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   SYS__STRING__NODES_MIN)                                            \
    /* ⛳ A DICTIONARY IS A HEAD OVER A TREE, and a program may make an empty one. What the tree is made  */ \
    /*   of is scaffolding a program cannot ask for: routing nodes last as long as their dictionary, and  */ \
    /*   a leaf is an array granule carved where list and stack chunks are.                              */ \
    X(PKG, SYS__KIND__DICTIONARY,       sys__dictionary__zzpackage_release_internal,                         \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__dictionary__zzpackage_construct, SYS__DICTIONARY__NODES)                                          \
    X(PKG, SYS__KIND__TREE_ROUTING_NODE, sys__tree_routing_node__zzpackage_release_internal,                 \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   SYS__TREE_ROUTING_NODE__NODES)                                     \
    X(PKG, SYS__KIND__DICTIONARY_LEAF,  sys__dictionary_leaf__zzpackage_release_internal,                    \
      SYS__COMPUTING_BASE__ARRAYS,       sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   SYS__DICTIONARY_LEAF__NODES)                                       \
    /* ⛳ ONE OF TWO ROWS WHOSE SIZE COLUMN IS A FLOOR RATHER THAN AN ANSWER, with the string's. This   */ \
    /*   one is asked for its extent, so what the column states is the SMALLEST an array can be — a      */ \
    /*   head and one element — and a caller wanting more says so when it asks.                          */ \
    /* ⛳ AND IT CANNOT BE COPIED YET, WHICH IS A GAP AND IS SAID TO BE ONE. A copy is expressible,     */ \
    /*   because an array owns what it holds — carve the same length, and take a hold for every element */ \
    /*   the copy names — and nothing has asked for one, so the row refuses rather than shipping a walk */ \
    /*   that no caller has ever run.                                                                   */ \
    /* ⛳ A PROCEDURE IS A NODE ARRAY OF TWO AND BORROWS THE ARRAY'S TEARDOWN WHOLE — that verb never   */\
    /*   asks what kind it was handed, because what kind it is, is how the drain reached it. What the   */\
    /*   row adds is a SIZE that is an answer rather than a floor, and a REFUSAL: a procedure made out  */\
    /*   of an arbitrary pair is a thing the evaluator will try to call, and a LENGTH cannot tell one    */\
    /*   from a real procedure. ⛳ `defun` IS THE DOOR, AND IT CHECKS BOTH THE CONTAINERS AND WHAT IS IN  */\
    /*   THEM: the parameters must be a quoted list or nothing and every parameter a quoted name, the    */\
    /*   body must be a quoted list. ▶ `procedure.cuh`.                                                  */\
    X(PKG, SYS__KIND__PROCEDURE,        sys__node_array__zzpackage_release_internal,                   \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                             \
      sys__heap_object__create_refusal,  SYS__PROCEDURE__NODES)                                        \
    X(PKG, SYS__KIND__NODE_ARRAY,       sys__node_array__zzpackage_release_internal,                         \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                               \
      sys__node_array__zzpackage_construct, SYS__NODE_ARRAY__NODES_MIN)                                  \
    /* ⛳ A HEADER OVER TWO TABLES, so its row is two nodes like every other header here and the size of  */ \
    /*   what it names is somebody else's row. ⛔ AND IT CANNOT BE COPIED YET: a copy would have to       */ \
    /*   decide whether the new environment shares the base or gets its own, and that is a question only  */ \
    /*   a caller can answer — `create` already takes the base, so the caller says it there instead.      */ \
    X(PKG, SYS__KIND__BINDINGS,         sys__bindings__zzpackage_release_internal,                          \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                                    \
      sys__bindings__zzpackage_construct, SYS__BINDINGS__NODES)                                              \
    X(PKG, SYS__KIND__LIST,             sys__list__zzpackage_release_internal,                               \
      SYS__COMPUTING_BASE__OBJECTS,      sys__heap_object__clone__default,                                    \
      sys__list__zzpackage_construct,    SYS__LIST__NODES)                                                   \
    X(PKG, SYS__KIND__LIST_CHUNK,       sys__list_chunk__zzpackage_release_internal,                        \
      SYS__COMPUTING_BASE__ARRAYS,       sys__heap_object__clone__default,                                    \
      sys__heap_object__create_refusal,   SYS__LIST_CHUNK__ALLOC)                                             \
    /* ⛳ A VIEW IS CONSTRUCTED ONLY BY THE LIST THAT IT VIEWS — there is no store for a program to name  */ \
    /*   as an argument, so `cdr` is how one is asked for and the row refuses.                           */ \
    /* ⭐⭐ THE ONE ROW WHOSE CLONE COLUMN IS NOT THE UNIVERSAL REFUSAL, AND IT IS THE *VIEW*, NOT   */ \
    /*   THE STORE. ⚖ *"wire deep_clone into the list row"* — and `sys__list__create` makes a      */ \
    /*   `SYS__KIND__LIST` store and hands back a VIEW, so what a program ever holds and would ask  */ \
    /*   to clone is a SUBLIST. Wiring the store row instead compiles, passes the kind census and   */ \
    /*   never fires — ⛔ `MEASURED`: the verb went on answering nothing. ⇒ ★ A ROW'S NAME IS       */ \
    /*   THE KIND'S, NOT THE WORD A PROGRAM USES FOR IT.                                            */ \
    /*   `sys__sublist__deep_clone` keeps a source->copy map, so SHARING AND CYCLES fall out rather */ \
    /*   than needing cases. ⛔ THAT IS WHAT THE FREEZE/THAW ROUTE CANNOT DO: it refuses a list met  */ \
    /*   twice with `SYS__LIST__FAULT_SHARED`, because the way back keeps no map.                   */ \
    /*   ⛳ THE STORE ROW STAYS A REFUSAL ON PURPOSE: a bare store with no view is not a thing a     */ \
    /*   program can name, and claiming a capability there would be claiming one nothing can reach. */ \
    X(PKG, SYS__KIND__SUBLIST,          sys__sublist__zzpackage_release_internal,                           \
      SYS__COMPUTING_BASE__OBJECTS,      sys__list__zzpackage_clone_object,                                   \
      sys__heap_object__create_refusal,   SYS__SUBLIST__NODES)                                                \
    X(PKG, SYS__KIND__CHUNK_PROVIDER,   sys__heap_chunk__zzpackage_release_internal,                        \
      SYS__COMPUTING_BASE__NO_PROVIDER,  sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal,   0u)

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH */
