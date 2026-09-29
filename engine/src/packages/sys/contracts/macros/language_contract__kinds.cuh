#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH

/* This file needs nothing: a macro body names nothing until something expands it. */
/* ── WHAT THIS PACKAGE'S KINDS ARE CALLED ────────────────────────────────────────────────────────────
 *
 * One row per kind, and the row is its IDENTITY: a number this package chooses and a name everybody
 * else uses. Nothing here says what a kind DOES — that is the objects list, in
 * `contracts/macros/language_contract__objects.cuh`, and the two are deliberately not one list.
 *
 * ⭐⭐ WHICH IS THE SEPARATION WORTH READING BEFORE THE ROWS: EVERY KIND HAS A ROW HERE, AND ONLY THE
 * ALLOCATED ONES HAVE A ROW BELOW. This package has thirty-five of the first and sixteen of the second.
 * The nineteen without are cell tags and values — an integer in a slot, a name mentioned, a quote, the
 * two truth objects, the answer to a question about a thing that is not there — and the two marks the
 * allocator stamps on a chunk. None of them is ever
 * carved, so none has a release, a provider or a size, and giving them empty columns in one merged list
 * would ask a reader to tell a kind that refuses to be built from a kind that is not built at all.
 * ⇒ ★ A KIND IS A NAME AND A NUMBER FIRST; BEING ALLOCATABLE IS SOMETHING SOME OF THEM ALSO ARE.
 *
 * ⛳ TWO COLUMNS AFTER THE PACKAGE AND THE NUMBER, AND ONE IS THE OTHER SAID IN THE PUBLISHED CASE:
 *
 *     spelling    what the interface publishes, and therefore what a program's author TYPES. The host
 *                 binding hands it over as `kinds()`, the composer resolves every kind a test writes
 *                 through that map, and a suite works by name and re-reads it every run.
 *     constant    what an engine author reads and compares in this tree. It is generated from the row,
 *                 tagged with the package, so a kind cannot be named one thing and dispatched as
 *                 another.
 *
 * ⭐⭐ THE SPELLING CARRIES THE PACKAGE AND THAT IS WHAT MAKES THE PUBLISHED MAP SAFE. `kinds()` is one
 * flat table across every package, built by assigning each name in turn — so two packages choosing one
 * spelling do not collide loudly, THE SECOND SILENTLY REPLACES THE FIRST and a kind that was published
 * yesterday answers somebody else's number today. Packed ids make a NUMBER collision structurally
 * impossible and nothing did the same for a NAME; a prefix does, without a check to run or remember.
 * ⛳ AND IT IS WHAT THE VERBS ALREADY DO — `sys__add`, `sys__system_register__get` — so this is the kinds
 * catching up with the words rather than a new convention.
 * ⛔ NOTHING SAYS `kind` IN IT, for the reason the verb list gives for not saying `verb`: a word
 * repeating a classification the position already states carries no information a reader did not have.
 *
 * ⛔⛔ SO A RENAME WORTH MAKING IS WORTH MAKING IN BOTH COLUMNS, TOGETHER, IN ONE ACT. A reader who meets
 * a different word on each side has to learn that they are one thing, and the two are one word by
 * construction: the spelling is the constant's own tail, lowercased, behind this package's name. That is
 * checkable rather than a habit, and it is checked.
 * ⛳ WHAT MAKES IT SAFE TODAY is that nothing outside this build has ever held either one: the map is
 * asked for at run time and a frozen picture carries the NUMBER. ⛔ THAT ENDS ON THE DAY A WHEEL SHIPS —
 * a caller's source then holds these spellings, and a name changed under it is a break with no compile
 * error anywhere, a dictionary lookup that starts answering `KeyError`. From the first release the left
 * column stays editable and the published one moves only with a deprecation.
 *
 * ⛔⛔ ZERO IS `STANDARD`, AND NOT BECAUSE UNWRITTEN ROOM READS IT — NOTHING CLEARS A NODE'S KIND. The
 * boot zeroes the counters, the fault channel and the reported word, and hands the allocation pool over
 * exactly as it was; publishing then writes ONE word per chunk and says so itself. A claimed chunk
 * therefore arrives holding whatever its last tenant left, and any kind at all can turn up in room
 * nobody wrote, zero among them.
 * ⇒ ★ EMPTINESS IS WRITTEN, NEVER READ OUT OF BARE ROOM. A register row answers `VALUE_NULL` because the
 * placing puts it there, so a row nobody has written and a row deliberately emptied are ONE state and
 * not two. Which is why no kind can be asked to mean "nobody has been here", and no numbering would make
 * one: that question is answered by what a maker WROTE.
 * ⛔ AND `INVALID` IS DELIBERATELY NOT ZERO. A verb that answers it has been ASKED SOMETHING WRONG, which
 * is a different sentence from "nobody has written here yet", and one number cannot say both.
 *
 * ── HOW THE NUMBERS ARE CHOSEN, WHICH IS LESS THAN IT LOOKS ──────────────────────────────────────────
 * They are this package's own and only have to be unique WITHIN it. What makes one unique everywhere is
 * the package id, pasted into the top byte at the gather — so a row never carries it and there is nothing
 * here anybody could forget to write. `sys` is package zero, so these pack to the values they already
 * had; the tag is there and it is nothing.
 * ⭐ GAPS ARE FINE AND NOTHING INDEXES AN ARRAY BY A KIND, so a sparse numbering costs nothing.
 * ⛔⛔ AND THE OTHER HALF IS THE ONE THAT BITES: A NUMBER YOU RETIRE MAY NEVER BE REUSED. A kind is
 * stamped into a node, and the day a node outlives the process that made it, a number handed to
 * something new is read as the old thing by every record still carrying it. These numbers are dense from
 * zero today and are NOT the ones the old tree used — its holes are ITS history and do not travel,
 * because nothing outside this build has ever held one. That freedom ends the moment a program is saved.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

#define sys__LANGUAGE_CONTRACT__KINDS(X, PKG)                                                          \
    /* An executable form with nothing said about it, which is also what memory nobody has written   */ \
    /* looks like. Held at zero by the guard beside the rows this list feeds.                        */ \
    X(PKG,  0, "sys__standard",            SYS__KIND__STANDARD)                                             \
                                                                                                       \
    /* ── WHAT A PROGRAM IS WRITTEN OUT OF ──────────────────────────────────────────────────────── */ \
    /* A name mentioned, a name quoted, and the two things a reader produces when it meets a quote. */ \
    X(PKG,  1, "sys__binding_reference",   SYS__KIND__BINDING_REFERENCE)                                    \
    X(PKG,  2, "sys__quoted_name",         SYS__KIND__QUOTED_NAME)                                          \
    X(PKG,  3, "sys__quoted_list",         SYS__KIND__QUOTED_LIST)                                          \
    X(PKG,  4, "sys__quoted_array",        SYS__KIND__QUOTED_ARRAY)                                         \
    /* ⚖ AND A NAME CAN BE SPELLED INSTEAD OF NUMBERED: *"the overflow bindings dictionary is keyed */ \
    /* on the heap node string object reference, not by number."* The cell holds the INTERNED       */ \
    /* string, so one text is one name, and takes no hold on it, because an interned string never   */ \
    /* dies. Which of the two a name is decides where it is read: a number is an array cell, a      */ \
    /* string is the dictionary behind the hatch. Their numbers are the next free ones, like the    */ \
    /* objects' below.                                                                              */ \
    X(PKG, 31, "sys__string_binding_reference", SYS__KIND__STRING_BINDING_REFERENCE)                        \
    X(PKG, 32, "sys__quoted_string_name",  SYS__KIND__QUOTED_STRING_NAME)                                   \
    /* ⭐⭐ A SUB-FORM THAT HAS NOT BEEN MADE YET. It names the ARRAY inside the picture that the form  */ \
    /* would be thawed from, and the evaluator turns it into a list at the moment it descends —       */ \
    /* which for a form that is never reached (the arm of an `if`) is never.                          */ \
    /* ⚖ architect: *"unthaw could be made to look at a binding for the list instead of instantiating */ \
    /* one"* — this is that, as a TAG rather than a verb, so nothing is dispatched and no apply is     */ \
    /* spent: `first_form` already reads the tag of every cell it steps over.                         */ \
    /* ⛔ ONLY A PLAIN `OBJECT_REFERENCE` SUB-ARRAY IS EVER FROZEN. A `QUOTED_LIST` is DATA handed to  */ \
    /* a verb, so it is thawed eagerly and no verb has to learn a new tag.                            */ \
    X(PKG, 34, "sys__frozen_list",         SYS__KIND__FROZEN_LIST)                                          \
                                                                                                       \
    /* ── WHAT A CELL CAN HOLD WITHOUT NAMING ANYTHING ──────────────────────────────────────────── */ \
    /* ⛳ AND THERE IS A TAG FOR TRUE AS WELL AS FOR FALSE, WHICH SCHEME DOES NOT NEED AND THIS DOES.*/ \
    /* A predicate here answers one of two objects and nothing else is a truth value, so the pair is*/ \
    /* the type rather than an economy — an integer handed to a test is a refusal and not a `#t`.   */ \
    X(PKG,  5, "sys__value_null",          SYS__KIND__VALUE_NULL)                                           \
    X(PKG,  6, "sys__value_true",          SYS__KIND__VALUE_TRUE)                                           \
    X(PKG,  7, "sys__value_false",         SYS__KIND__VALUE_FALSE)                                          \
    X(PKG,  8, "sys__value_int",           SYS__KIND__VALUE_INT)                                            \
    /* ⭐⭐⭐ A REAL NUMBER, AND IT IS `sys`'s RATHER THAN `nn`'s — ⚖ RULED: *"i think float */ \
    /* as a word should exist … i do wonder if it should be in nn or sys, probably sys."* ⛳ THE AXIS  */ \
    /* DECIDES IT: `sys` is the language you write IN. If `nn` owned floats then `sys__add` could not */ \
    /* add two of them, and `nn` would need its own arithmetic — the language, twice.                 */ \
    /* ⛳ IT IS fp64, AND THAT COSTS NOTHING: a cell's argument is already a `uint64_t`, so a double   */ \
    /* fits one word exactly with five to spare. ⚖ *"it is 64B so there is plenty of space."*         */ \
    /* ⛔ THE BITS LIVE IN `args[0]` AS A REINTERPRETATION, NOT A CONVERSION. Nothing rounds on the    */ \
    /* way in or out, so a float that crosses the seam and comes back is the same number.             */ \
    X(PKG, 33, "sys__value_float",         SYS__KIND__VALUE_FLOAT)                                          \
    /* ⛔ AND ONE OF THESE IS A FAILURE THAT COULD NOT AFFORD AN OBJECT. It carries the fault word  */ \
    /* in the cell itself, which is what makes it usable on the one path that failed BECAUSE it     */ \
    /* could not allocate — see `ERROR` below, the same failure when there WAS room. Neither        */ \
    /* reference set admits this tag, so nothing ever follows its first argument as an offset.      */ \
    X(PKG,  9, "sys__value_error",         SYS__KIND__VALUE_ERROR)                                          \
                                                                                                       \
    /* ── AND WHAT A CELL HOLDS WHEN IT NAMES SOMETHING ─────────────────────────────────────────── */ \
    /* A reference, and the one kind of reference that says what is at the other end. ⛔ BOTH ARE    */ \
    /* CELL TAGS AND NEITHER IS AN OBJECT'S: the head's kind says what a thing IS, and these say    */ \
    /* what a slot HOLDS. A procedure reference names a procedure; the two are never the same word  */ \
    /* in the same field, which is the whole reason they are two rows.                              */ \
    X(PKG, 10, "sys__object_reference",    SYS__KIND__OBJECT_REFERENCE)                                     \
    X(PKG, 11, "sys__procedure_reference", SYS__KIND__PROCEDURE_REFERENCE)                                  \
                                                                                                       \
    /* ── WHAT THE LANGUAGE'S OBJECTS ARE ───────────────────────────────────────────────────────── */ \
    /* Every one of these has a row in `contracts/macros/language_contract__objects.cuh` saying how it is        */ \
    /* carved, released and copied.                                                                 */ \
    /* A kind here with no row there is refused at creation rather than built wrongly.              */ \
    X(PKG, 12, "sys__list",                SYS__KIND__LIST)                                                 \
    X(PKG, 13, "sys__list_chunk",          SYS__KIND__LIST_CHUNK)                                           \
    X(PKG, 14, "sys__sublist",             SYS__KIND__SUBLIST)                                              \
    X(PKG, 15, "sys__stack",               SYS__KIND__STACK)                                                \
    X(PKG, 16, "sys__stack_chunk",         SYS__KIND__STACK_CHUNK)                                          \
    X(PKG, 17, "sys__node_array",          SYS__KIND__NODE_ARRAY)                                           \
    X(PKG, 18, "sys__bindings",            SYS__KIND__BINDINGS)                                             \
    /* ⛳ A PROCEDURE IS A NODE ARRAY OF TWO AND IS NOT ONE. The room and the teardown belong to the */ \
    /* array; the kind is what lets `type` answer for the thing itself, so a pair BUILT out of a    */ \
    /* list is not a thing to run and an array of two cannot become one by being shaped like it.    */ \
    X(PKG, 19, "sys__procedure",           SYS__KIND__PROCEDURE)                                            \
    /* ⛳ AND AN ERROR IS AN OBJECT, WHICH IS WHY IT SITS HERE and not with the values: it is carved,*/ \
    /* counted, held and released like any other, and a cell names it with an object reference.     */ \
    X(PKG, 20, "sys__error",               SYS__KIND__ERROR)                                                \
    X(PKG, 21, "sys__heap_object",         SYS__KIND__HEAP_OBJECT)                                          \
    X(PKG, 22, "sys__computing_base",      SYS__KIND__COMPUTING_BASE)                                       \
    /* ⛳ TEXT, AND ITS NUMBER IS THE NEXT FREE ONE RATHER THAN ITS PLACE. Nothing orders these rows by  */ \
    /* number, and moving `INVALID` to make room would change a published value for nothing.          */ \
    X(PKG, 27, "sys__string",              SYS__KIND__STRING)                                               \
    /* A dictionary, and the two kinds its tree is made of: a routing node, and a leaf of seven pairs. */ \
    X(PKG, 28, "sys__dictionary",          SYS__KIND__DICTIONARY)                                           \
    X(PKG, 29, "sys__tree_routing_node",   SYS__KIND__TREE_ROUTING_NODE)                                    \
    X(PKG, 30, "sys__dictionary_leaf",     SYS__KIND__DICTIONARY_LEAF)                                      \
                                                                                                       \
    /* ── WHAT THE ALLOCATOR SAYS ABOUT ITS OWN ROOM ────────────────────────────────────────────── */ \
    /* These are stamped into the first node of a chunk and never into anything a program holds. A  */ \
    /* chunk is a kind that is NEVER CARVED: it is the room, so its row asks for no space at all.   */ \
    X(PKG, 23, "sys__heap_chunk",          SYS__KIND__HEAP_CHUNK)                                           \
    X(PKG, 24, "sys__heap_chunk_active",   SYS__KIND__HEAP_CHUNK_ACTIVE)                                    \
    X(PKG, 25, "sys__chunk_provider",      SYS__KIND__CHUNK_PROVIDER)                                       \
                                                                                                       \
    /* ── AND THE ONE THAT IS NOT A KIND ────────────────────────────────────────────────────────── */ \
    /* What a question about a thing that is not there answers.                                     */ \
    X(PKG, 26, "sys__invalid",             SYS__KIND__INVALID)

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_MACROS_LANGUAGE_CONTRACT__KINDS_CUH */
