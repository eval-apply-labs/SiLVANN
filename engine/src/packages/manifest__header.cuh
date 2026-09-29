#ifndef SILVANN__PACKAGES_MANIFEST__HEADER_CUH
#define SILVANN__PACKAGES_MANIFEST__HEADER_CUH
/* What this file needs: the fixed-width integers its package ids are written in. A HIP compile gets
 * them from the vendor's runtime header it includes first and a CUDA one does not, so the file asks for
 * them itself. */
#include <stdint.h>
/* ══ WHAT A PACKAGE OWES ═════════════════════════════════════════════════════════════════════════════
 *
 * This file is the contract between packages and the machinery that gathers them, and it is the half a
 * package COMPILES AGAINST — the gathering itself is next door, in the two files generated from the roster
 * below: `manifest.cuh` for the program and `manifest__gpu.cuh` for a silicon family's binary. It is at the root of
 * `packages/` and belongs to no package, which is the point: a package must be able to learn what it owes
 * WITHOUT READING ANOTHER PACKAGE.
 *
 * ⚖ ARCHITECT: *"my worry is that this should then be outside of sys ... ai_silvann must not touch sys.
 * so either the manifest gathers the macro from the various packages or they add their own in a file."*
 * They add their own, and the worry is answered by WHERE the file sits rather than by which mechanism
 * gathers it — see below. What was genuinely missing is this document: the rule lived in a comment inside
 * `sys/cpu/heap_object__impl.cuh`, so a second package could not find out what it owed without opening `sys`.
 *
 * ⭐⭐ AND WHAT A PACKAGE PUBLISHES IS BROADER THAN ITS OBJECTS, WHICH THE NAMING NOW ANTICIPATES. ⚖
 * ARCHITECT: *"this is what publishes objects that will eventually be callable in LISP ... maybe one
 * macro for the objects and one for the functions? ... which is what each package will send to the ast as
 * a contract to publish its own words into the language. this might even contain structures and enums."*
 *
 * ⭐ SO THE GROUPING IS ON THE CONTRACT AND NOT ON THE CATEGORY. ⚖ ARCHITECT: *"why not a single file? if
 * you dont want a single file at least it should be x__LANGUAGE_CONTRACT__OBJECTS etc so the grouping is on
 * the ast contract end."* Right, and it is the same rule as `_from_head` being a suffix: put the COMMON
 * part where it groups and the VARYING part where it varies. `sys__LANGUAGE_CONTRACT__*` shows everything a
 * package publishes under one prefix; `..._AST_OBJECT__REGISTRATION_LIST` scattered them by category and
 * grouped them by mechanism, which nobody searches for.
 *
 *   ONE FILE   `<package>/language_contract.cuh`, which includes one file per list from the package's
 *              `contracts/macros/` — `language_contract__objects.cuh`, `__verbs.cuh`, `__kinds.cuh`:
 *
 *     <package>__LANGUAGE_CONTRACT__OBJECTS     BUILT — the kinds this package registers, below.
 *     <package>__LANGUAGE_CONTRACT__VERBS   BUILT — the words it publishes into the language: one
 *                                          row per verb, carrying a spelling, an apply and a BASE the
 *                                          gather tags. FOUR consumers expand it — the verb dispatch in
 *                                          `sys/cpu/heap_object__impl.cuh`, the host's vocabulary table and
 *                                          the package-init gather in `engine/abi.cuh`, and the verb-id
 *                                          constants in `language_contract_verbs.cuh`. ⛔ WHICH IS WHY A
 *                                          PACKAGE WITH NO VERBS STILL DEFINES AN EMPTY ONE: the macro name
 *                                          is PASTED from the row, so all four name it whether or not it has
 *                                          anything to say.
 *     <package>__LANGUAGE_CONTRACT__KINDS       BUILT — what this package's kinds are CALLED: one row per
 *                                          kind, carrying a spelling and a BASE the gather tags. TWO
 *                                          consumers expand it — the tagged constants in
 *                                          `language_contract_kinds.cuh` and the host's type table in
 *                                          `engine/abi.cuh`. ⛔ IT IS A KIND'S IDENTITY AND NOT ITS
 *                                          BEHAVIOUR: every kind has a row here, and only the ALLOCATED
 *                                          ones have one in OBJECTS — the claim gate's
 *                                          `kind_row_has_contract_row` reports both counts — and merging
 *                                          the lists would ask a reader to tell a kind that REFUSES to be
 *                                          built from one that is never built at all.
 *
 *   AND ONE THING THAT IS NOT A LIST BUT A SINGLE ENTRY POINT:
 *
 *     <package>__package__init            ✅ BUILT, AND IT IS A PUBLISHED VERB RATHER THAN A C ENTRY
 *                                          POINT. ⚖ ARCHITECT: *"init therefore takes the init of all
 *                                          the packages and builds it as a lisp procedure and executes
 *                                          it."* The boot stands the LANGUAGE up in C, because nothing
 *                                          can run before it; it then builds one call per package into a
 *                                          list and EVALUATES it, in a machine that is already whole.
 *                                          ⭐⭐ SO A ROW'S POSITION DECIDES NOTHING, WHICH IS THE
 *                                          OPPOSITE OF WHAT THIS PARAGRAPH SAID IT WOULD. It said a call
 *                                          in row order was the one thing a row's position would decide.
 *                                          It is not: an init that finds what it depends on still in the
 *                                          list puts a copy of itself at the tail and answers nothing, so
 *                                          the packages sort themselves and the roster's order is an
 *                                          include order and an id order and nothing more.
 *                                          ⇒ ★ WHICH IS WHAT MAKES ADDING A PACKAGE A LOCAL ACT. Nothing
 *                                          outside its own directory has to know where it belongs in a
 *                                          sequence, because there is no sequence to belong in.
 *                                          ⛳ AND THE READINESS STATE IS THE LIST ITSELF — a call still in
 *                                          it has not run, one that has is gone — so nothing publishes a
 *                                          flag and nothing reads one. ▶ `sys/cpu/package__header.cuh`.
 *
 * ⛳ IT IS FOR STORAGE A PACKAGE OWNS AND THE ENGINE DOES NOT. Everything carved from the heap needs no
 * init — the allocator is already standing when a package's first line runs. What needs one is memory
 * that was never the heap's: an arena, a table, a cache, anything a node in the tree can NAME while its
 * bytes live somewhere else entirely. Only the package knows what that is, and the engine learns nothing
 * by calling it.
 * ⛔ A PACKAGE WITH NOTHING TO BUILD STILL DEFINES AN EMPTY ONE, for the reason every uniform list has:
 * the caller is generated, and a generated caller that has to know which packages opted in is a caller
 * written by hand again. ⚠ AND FOR `package_init` THE COMPILER DOES NOT CATCH THE OMISSION: the only
 * generated per-row checks are the two `static_assert`s on `PACKAGE_<name>_HEADER_PRESENT` and
 * `PACKAGE_<name>_PRESENT` in `packages/manifest.cuh`, and neither asks about an init. An empty
 * definition becomes its own presence check — a package that reserves an id and ships none failing
 * to compile AT ITS OWN ROW, naming itself — on the day a generated caller pastes the name.
 *
 * ⛔ SEPARATE LISTS RATHER THAN ONE MACRO WITH SEVERAL CALLBACKS, and the reason is who pays when a
 * category is added. One `AST_CONTRACT(OBJECT, FUNCTION, TYPE)` works, but every consumer must then pass
 * a callback for every category even to use one, and adding a fourth edits every expansion site in the
 * tree. Separate lists cost a second macro in the file that already exists and change nothing anywhere
 * else. ⇒ ★ THE FILE IS THE CONTRACT; THE LISTS ARE ITS SECTIONS. That gives one place to look, which is
 * what "one contract per package" is actually asking for, without making the mechanism pay for it.
 *
 * ── ONE THREAD PER BLOCK, WHICH IS THE CONDITION EVERYTHING HERE RUNS UNDER ─────────────────────────
 * ⚖ **RULED:** *"everything but the status will be dealt in a single threaded way."* Every verb in every
 * package is called by ONE thread of a block; the exceptions are the words that swap, and they say so.
 * The election is `threadIdx.x == 0`, IN EVERY BLOCK: each kernel in `engine/abi.cuh` returns every lane
 * but thread zero and then branches on `sys__silicon__block_id()`, so a package verb runs on one thread
 * of ANY block. It is the engine's convention and it happens ABOVE the packages — so there is nothing
 * here to enforce and no guard to look for.
 * ⛔ WHAT IT IS NOT IS AN ELECTION OF ONE BLOCK. `block_id() != 0u || threadIdx.x != 0u` is the BOOT's,
 * for the three steps that must happen once for the whole launch (`engine/boot.cuh`). Everywhere else
 * block zero is one block among many: `k_reside` parks it in `await_instructions` alongside the rest,
 * `k_dispatch` and `k_grid` put package work on the blocks that are NOT zero, and `start_blocks` runs
 * from thread zero of every block. ⇒ ★ SINGLE-THREADED IS A CLAIM ABOUT A BLOCK'S LANES AND NOT ABOUT
 * THE GRID. Device-global state a package owns is reached by as many threads as there are blocks, and
 * reading this line as "block zero only" is how such state comes to be treated as uncontended.
 * ⛳ AND THE BLOCK ID IS ASKED FOR RATHER THAN READ: `sys__silicon__block_id()` is the only spelling
 * the engine uses, because `blockIdx` is a backend's word and lives behind the silicon seam.
 *
 * ⛳ IT IS STATED HERE BECAUSE IT IS A PRECONDITION ON CALLERS AND HAD NO HOME. The silicon seam tells a
 * BACKEND that its primitives are called single-threaded; `heap_object__header.cuh` tells a caller that
 * lane-level exclusion is its own debt. Neither told a caller of a package that the whole package assumes
 * it. ⇒ ★ A CONDITION THAT HOLDS EVERYWHERE IS THE EASIEST ONE TO LEAVE UNWRITTEN, and then it reads as a
 * missing guard in whichever function happens to mention it — which is exactly what happened to
 * `sys__heap__zzprivate_claim_chunk`, where breaking it is most expensive and so most likely to be said.
 *
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE PACKAGES THEMSELVES ─────────────────────────────────────────────────────────────────────────
 * One row per package: the id it keeps forever, and the name everything else is pasted from. Nothing that
 * CONSUMES this list is edited when a package is added — every consumer takes a row and reads the column
 * it needs.
 * ⭐⭐ AND NOBODY WRITES A ROW. ⚖ *"if it can be incremental instead it is worth doing"*: a package is a
 *   folder of `packages/` holding a `manifest.cuh`, and every build (`scripts/src_enroll.py`, run by the
 *   assembler) APPENDS a folder not yet listed, with the next id. The list is committed, and the tool only
 *   ever appends to it — ids are never renumbered or reused, and a listed folder that is gone stops the build
 *   by name. Two branches that each add a package collide on the same line, which git shows.
 *
 * ⭐ THREE COLUMNS, AND THE THIRD IS A DIFFERENT QUESTION FROM THE SECOND. The NAME is the identity: it
 * is pasted onto every contract macro and onto the id, it is written into saved state through the top
 * byte of a kind, and it can never move. The PATH is where that identity currently lives, and it can.
 * ⚖ ARCHITECT: *"it might be good if we end up in a situation like this where we are subbing out a
 * package for a v2, or to have different version numbers match the same prefix."* Substituting an
 * implementation is then editing one column, and nothing that was compiled against the package notices —
 * which is exactly the property a name that everything pastes from cannot have.
 *
 * ⛳ THE PATH IS WRITTEN ONCE, WHICH IS WHAT MAKES THE COLUMN SAFE. The assembly is generated from this
 * row, so a path exists in exactly one place and a wrong one is a file the compiler cannot open. A path
 * a reader copied by hand into an `#include` would be a second copy of the same fact with nothing
 * comparing them. ⇒ ★ A COLUMN IS ONLY AS DANGEROUS AS THE NUMBER OF PLACES IT IS COPIED TO.
 *
 * ⛳ AND IT GOES LAST ON PURPOSE, SO THAT APPENDING HERE IS FREE FROM NOW ON. A consumer absorbs the
 * columns it does not read by taking `...` for them, which needs a tail to hold — and while the name was
 * last there was none, so every adapter had to name every column and a fourth would have edited them all.
 * The path is now the tail. ⇒ ★ THE CONSUMER OF THE LAST COLUMN PAYS, so the last column should be the
 * one nobody reads.
 *
 * ⛔ THE ID IS PERMANENT. It is the top byte of every kind this package registers, and a kind is written
 * into saved state — so an id, once issued, is never renumbered and never reused, for the same reason a
 * retired kind's number stays a gap.
 * ⚠ THE OLD TREE KEEPS A REGISTRY OF ITS OWN, for its own build; nothing here reads it (the guard below
 * says how that is known). This is the row set this tree is written against.
 * ⛳ AND A SUBSTITUTED PACKAGE KEEPS ITS ROW: pointing a name's path at `nn_v2` leaves `nn/` on disk, and the
 *   enrolment skips a folder whose name is already a row's, so the old folder is not enrolled a second
 *   time under the same name. */
#define PACKAGE_LIST(X) \
    X(0, sys, "sys") \
    X(1, nn, "nn") \
    X(2, ai_qwen_3, "ai_qwen_3") \
    X(3, ai_glm_5_3, "ai_glm_5_3")

/* The ids, generated, so that nobody writes a number twice.
 *
 * ⛔ AND GUARDED AGAINST A SECOND REGISTRY THIS TRANSLATION UNIT DOES NOT CONTAIN. The tree this one grew
 * out of has a registry of its own, with a different row shape, and both name a package `sys` at id
 * zero — so both would generate `sys_pkg_id`, and the second one is a REDEFINITION rather than an
 * agreement: a `constexpr` is not a macro, and two identical spellings of one do not merge the way two
 * identical `#define`s do. Where the two are composed, whichever is read first generates and the other
 * sees the flag and stands down, and the value is the same either way because the row is the same.
 * ⛳ NOTHING COMPOSES THEM HERE. `MEASURED`: `grep -rn '#include' src/ | grep 'src_old/'` is EMPTY — no
 * file reachable from this one names the old tree, so the flag has no writer this build can meet and
 * the branch below is always taken. That is the same condition a host build of the package alone runs
 * under, which is what keeps the harness working with no arrangement of its own.
 * ⛳ RETIREMENT: the guard's subject is COMPOSITION rather than the rename, so it is already subjectless
 * and goes with the line that sets the flag over there. Renaming the package to dodge the collision
 * would mean renaming it twice, and the second rename is the one that touches every name in the tree. */
#ifndef SILVANN_PACKAGE_IDS_PRESENT
#define PACKAGE_ID_DEF(id, name, ...) constexpr uint64_t name##_pkg_id = (id);
PACKAGE_LIST(PACKAGE_ID_DEF)
#undef PACKAGE_ID_DEF
#define SILVANN_PACKAGE_IDS_PRESENT 1
#endif

/* How many there are, DERIVED by expanding the same list with an arm that reads nothing and contributes
 * one. Nobody types it, so it cannot disagree with the roster — and a number that says how many packages
 * exist is exactly the kind that would otherwise be updated one commit late.
 * ⛳ ITS READERS SIZE THINGS ONE PER PACKAGE: the engine's rooms, the init's turns and the extension
 *   hatch (`engine/abi.cuh`, `sys/cpu/package__impl.cuh`), and a silicon family's table of door tables
 *   (`silicon_families/entry.cuh`). `grep -rn package_count` over `src/` lists them. */
#define PACKAGE_COUNT_ARM(id, name, ...) + 1
constexpr uint64_t package_count = 0 PACKAGE_LIST(PACKAGE_COUNT_ARM);
#undef PACKAGE_COUNT_ARM

/* And the check that the sentence above is TRUE rather than hopeful. Generating the constants stops
 * anyone writing a number twice IN CODE; it does not stop two rows carrying the same number, because
 * each generates its own differently-named constant and the compiler has no reason to compare them.
 * A switch does compare them: duplicate labels are an error, and the diagnostic names the value.
 * ⛳ NOTHING CALLS THIS AND NOTHING SHOULD. It exists to be compiled, which is the whole of its job —
 * an id collision is unrecoverable rather than inconvenient, since a kind carries its package in the
 * top byte and is written into state that outlives the process. */
#define PACKAGE_ID_CASE(id, name, ...) case (id):
static inline void package_ids_are_unique(int which) {
    switch (which) { PACKAGE_LIST(PACKAGE_ID_CASE) default: break; }
}
#undef PACKAGE_ID_CASE

/* And the packing for the words a package publishes, which is the same arrangement as its kinds one
 * category over: a package numbers its verbs from zero and the machinery puts the package in the top
 * bits, so two packages cannot collide and neither writes the other's number.
 * ⛳ IT IS 64 BITS HERE AND 32 FOR A KIND, and the difference is not a preference: a kind has to fit the
 * four-byte field it is stamped into, and a verb id is never stored in a node. So the tag sits at bit 32
 * and a package has four billion verbs it will never use. */
/* And the same for a KIND, which is the other thing a package numbers from zero. The tag sits at bit 24
 * because the field is four bytes wide: one byte of package, three of kind.
 * ⛳ WRITTEN TO MATCH THE OTHER TREE'S SPELLING EXACTLY, AND NOTHING CHECKS THAT IT DOES. The two
 * registries sit in separate translation units — the same emptiness the id guard above rests on, and
 * measured by the same command — so a differing redefinition is never a diagnostic here: nothing sees
 * both spellings at once, and drift between them is silent. ⇒ ★ THE AGREEMENT IS HELD BY HAND, SO EDIT
 * BOTH SPELLINGS OR NEITHER. What makes it worth holding is what the macro packs: one byte of package
 * and three of kind is a claim about where a package's byte lives, and a kind is stamped into a node
 * and saved, so a value one tree writes is read back by whatever runs next. */
/* ⛔⛔ AND IT WAS APPLIED TWICE WHICH WAS HARMLESS ONLY WHILE THE PACKAGE WAS ZERO.
 * The KINDS gather defines each kind's constant ALREADY TAGGED (`language_contract_kinds.cuh`), and every
 * OBJECTS/heap arm then wrote `case PACKAGE_DTYPE(PKG, DTYPE):` over a row whose column holds that same
 * constant. `sys` is package 0, so `27 + 0` is 27 and nothing showed; `nn`'s first OBJECTS row would have
 * computed 0x01000000 + 0x01000000 and matched NOTHING — its release, clone, provider and size all falling
 * to their defaults, silently, for every object of that kind.
 * ✅ ⚖ RULED AND BUILT: THE ARMS STOP TAGGING AND A ROW NAMES THE PUBLISHED KIND. The tag is
 * applied exactly once, here, where the constant is defined. Five arms changed (`heap__impl.cuh` ×2,
 * `heap_object__impl.cuh` ×3) and two claim rules were rewritten to state the new invariant. ⛳ IT WAS NOT
 * THE PER-ROW TAGGING THE OPCODE LESSON WARNS ABOUT: that was a SWEEP over rows, and this stayed one
 * list-driven application.
 * ⭐ THE INVARIANT NOW, AND IT IS TWO DIFFERENT RULES BECAUSE THE TWO LISTS CARRY DIFFERENT THINGS:
 *     a VERBS row carries a BARE BASE          ⇒ its arms MUST tag   (`PACKAGE_VERB(PKG, BASE)`)
 *     an OBJECTS row names the PUBLISHED KIND      ⇒ its arms MUST NOT   (`case DTYPE:`)
 *   The tag is applied when the opcode list is gathered, not written into each row. That is not a style
 *   choice: a row can be forgotten, and a forgotten row is an untagged opcode that collides silently.
 *   A list cannot forget; a sweep can.
 *   The OBJECTS arms are now shaped like `ENG__ABI__KIND_ROW`/`ENG__ABI__VERB_ROW` in `engine/abi.cuh`,
 *   which have always published the generated `NAME` rather than re-deriving it from `BASE`.
 * ⛔⛔ AND THE GATE THAT EXISTS TO CATCH DOUBLE TAGGING COULD NOT SEE THIS ONE — `MEASURED`,
 *   and it is why the defect survived to be found by reading. `row_tag_at_gather` flags a row whose base
 *   is ALREADY PACKED, but it tests that with an INTEGER parse (`_as_int`, ceiling 1<<24). An OBJECTS row
 *   spells a NAME — `SYS__KIND__STACK` — so the parse returns nothing and the row passes as "a bare base"
 *   while holding a fully tagged constant. The rule reported `12 row(s) … no row carries a tag` and was
 *   right about every character it examined. ⇒ ★★ A CHECKER THAT RESOLVES ONLY LITERALS IS BLIND TO
 *   EXACTLY THE ROWS A CODEBASE WRITES BY NAME, and it returns a clean specific green while blind —
 *   `CLAUDE.md`'s false-negative family, arriving this time through a CONSTANT FOLD nobody performed. */
#define PACKAGE_DTYPE(pkg, base)  ((uint32_t)(base) + ((uint32_t)pkg##_pkg_id << 24))
#define PACKAGE_DTYPE_PKG(d)      ((uint32_t)(d) >> 24)
#define PACKAGE_DTYPE_BASE(d)     ((uint32_t)(d) & 0xffffffu)

#define PACKAGE_VERB(pkg, base)   ((uint64_t)(base) + ((uint64_t)pkg##_pkg_id << 32))
#define PACKAGE_VERB_PKG(v)       ((uint64_t)(v) >> 32)
#define PACKAGE_VERB_BASE(v)      ((uint64_t)(v) & 0xffffffffull)

/* ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ── THE OBJECT KIND REGISTRATION ────────────────────────────────────────────────────────────────────
 *
 *   WHERE   `<package>/contracts/macros/language_contract__objects.cuh`, in the package's OWN directory,
 *           included by its `language_contract.cuh`.
 *   WHAT    `<package>__LANGUAGE_CONTRACT__OBJECTS(X, PKG)`, spelled with the package's `PACKAGE_LIST` name.
 *           It takes the arm to expand AND the package to expand it for, and forwards the second into
 *           every row — which is how an arm learns whose kind it is holding without the package ever
 *           writing its own name in a row. ⛔ THE PACKAGE FORWARDS `PKG`, NEVER ITS OWN SPELLING: a row
 *           that names its package is a row that keeps being right after it is copied into another one.
 *   ROWS    one per kind, seven columns:
 *
 *               X(PKG, dtype, release_method, provider_slot, clone_method, construct_method, nodes)
 *
 *               PKG              forwarded, never typed. The gather supplies it from the roster.
 *               dtype            what the object says it is — the kind's PUBLISHED constant
 *                                (`SYS__KIND__STACK`), already tagged where the KINDS list defines it;
 *                                an arm never tags it again (the note below).
 *               release_method   what it hands over when its last reference goes. It HANDS OVER and
 *                                never releases — see rule ⑤ of the gate for why that is not a style
 *                                preference.
 *               provider_slot    which chunk it is carved from: `SYS__COMPUTING_BASE__OBJECTS` for what
 *                                LASTS, `..._ARRAYS` for what CHURNS, `..._NO_PROVIDER` for a kind that
 *                                is placed by hand and never made.
 *               clone_method     how it is copied, or `sys__heap_object__clone__default` to refuse.
 *               construct_method how it is built from the language, or `..._create_refusal` to refuse.
 *               nodes            how many nodes one of them occupies.
 *
 * ⭐⭐ NOBODY EDITS ANYBODY ELSE'S FILE, AND THAT IS THE WHOLE ARRANGEMENT. `sys` registers its kinds in
 * `sys/contracts/macros/language_contract__objects.cuh`; `nn` registers its own in its own directory, at
 * the same filename, defining `nn__LANGUAGE_CONTRACT__OBJECTS`. The dispatch pastes each package's name onto
 * `__LANGUAGE_CONTRACT__OBJECTS` and expands them all into one switch, written once and never edited as packages
 * arrive.
 * ⇒ ★ WHICH IS WHY THE LIST BEING INSIDE `sys/` IS NOT THE COUPLING IT LOOKS LIKE: it is not a shared
 * registry that happens to live in `sys`, it is SYS'S OWN ANSWER, and every package has one. The coupling
 * would be a single list in this file, which is the shape the project guide already rules out for the
 * dtype enum — one shared file means adding a kind by editing somebody else's package.
 *
 * ── HOW A PACKAGE NUMBERS ITS KINDS, WHICH IS THE ONE PART IT DOES FOR ITSELF ───────────────────────
 *
 *   ① NUMBER THEM FROM ZERO, IN YOUR OWN FILE, AND STOP AT THREE BYTES. A kind's number is yours to
 *      choose: 0, 1, 2, or 7 and 900 if that reads better. The bound is 16,777,215, which is four orders
 *      of magnitude past anything conceivable, and nothing outside your package ever sees these numbers.
 *
 *   ② YOU NEVER WRITE YOUR PACKAGE'S ID. It is written once, where packages are listed, and the
 *      machinery pastes it onto the top byte of every kind you register. A row therefore carries a BASE
 *      and never a packed value.
 *      ⛔ AND THAT IS A RULE RATHER THAN A CONVENIENCE, because the alternative was tried one category
 *      over and measured: applying the tag per row instead of once at the gather let 33 opcodes escape
 *      UNTAGGED — a collision inside the mechanism built to prevent collisions, and invisible, because
 *      dispatch matched on names. ⇒ ★ A LIST CANNOT FORGET; A SWEEP CAN.
 *
 *   ③ GAPS ARE FINE, AND AFTER A RETIREMENT A GAP IS REQUIRED. Nothing indexes an array by a kind and
 *      nothing tests a range of them — `MEASURED` across both trees: zero such sites, because every
 *      consumer compares for equality and the host binding is a list of name/value pairs rather than a
 *      table. So a sparse numbering costs nothing.
 *      ⛔⛔ AND THE OTHER HALF IS THE ONE THAT BITES: A NUMBER YOU RETIRE MAY NEVER BE REUSED. A kind is
 *      stamped into a node, and nodes are saved into conversations and written into logs, so a number is
 *      read back by something that was not running when it was written. Reusing one silently
 *      reinterprets every record that carries it. The same holds one level up: A PACKAGE ID, ONCE
 *      ISSUED, IS NEVER RENUMBERED AND NEVER REUSED — it is part of an identity that outlives the
 *      process, which is the whole difference between this and an opcode.
 *
 *   ④ AND THE TYPE THAT HOLDS A KIND HAS A FIXED WIDTH, WITHOUT WHICH NONE OF THE ABOVE IS
 *      REPRESENTABLE. `sys__kind` is a `uint32_t` and every kind is a `constexpr` of that width,
 *      generated from the rows rather than written. ⛔ IT WAS AN ENUM, AND AN ENUM CANNOT CARRY THIS: one
 *      with no stated underlying type has only the range its own members need, so a tagged kind falls
 *      outside it and a device compiler REFUSES IT OUTRIGHT. ⚠ `MEASURED`, and the reason it is worth
 *      keeping here after the fact: the host compiler accepts the same line silently, so a selftest can
 *      be green on something a device compiler will not take. Stating an underlying type answered that
 *      while ONE package wrote every row; a constant answers it for everybody and needs no guard to
 *      prove it did.
 *
 * ⛳ THE TAG SITS AT BIT 24, WHICH IS FORCED RATHER THAN CHOSEN: the field is four bytes, and the node it
 * sits in is a fixed size the host side indexes by. One byte of package, three of kind.
 * ⛳ AND `sys` IS PACKAGE ZERO, so its kinds pack to the values they already had. Nothing moved to make
 * this true, which is why it could be adopted without touching a single comparison in either tree.
 *
 * ✅ A SECOND PACKAGE NOW HAS A NAME AS WELL AS A NUMBER, WHICH IT DID NOT WHEN THE RULES ABOVE WERE
 * WRITTEN. Its kinds packed and dispatched correctly and nothing outside the device knew what to call
 * them, because the type names published to the host came from `sys`'s list alone — so a kind registered
 * anywhere else was unnameable from Python, and since every kind a test writes is resolved through that
 * map, the check that such a kind worked could not have been written either.
 * `<package>__LANGUAGE_CONTRACT__KINDS` is that list, and it GENERATES the tagged constants from the
 * bases as well as the host's table, which is what makes ② mechanical rather than remembered — one list,
 * a name and a number, and neither written twice.
 * ⛳ ITS GATHER RUNS IN A PHASE AHEAD OF EVERY HEADER, WHICH IS THE ONE PLACE THE KINDS DIFFER FROM THE
 * VERBS. A kind is compared inside DECLARATIONS — whether a node carries a reference, what an empty
 * register row answers — so its constants have to exist before the first header is read. A verb is only
 * ever compared inside an implementation, so its names can be gathered after the last one. Both gathers
 * sit beside this roster; they just sit on either side of the header phase.
 *
 * ── WHAT IS NOT BUILT ───────────────────────────────────────────────────────────────────────────────
 * ⛳ NOT THE GATHERS — THEY ARE ROSTER-DRIVEN, AND A SECOND PACKAGE EDITS NONE OF THEM. It belongs
 * under this heading anyway, because a gather written by hand is the first thing a reader comes here
 * looking for. `MEASURED`, and the command is part of the claim: `grep -rn 'GATHER(0, sys)'` over
 * `src` finds prose only — every kind gather expands `PACKAGE_LIST` instead (create, verb, release
 * and clone in `sys/cpu/heap_object__impl.cuh`; provider and nodes in `sys/cpu/heap__impl.cuh`), and so does
 * the host's verb table in `engine/abi.cuh`. ⇒ ★ WHICH IS THE ROSTER'S OWN PROPERTY, STATED ABOVE:
 * nothing that CONSUMES the list is edited when a row arrives.
 * ⚠ AND NO EMPTY DEFAULT EXISTS FOR A PACKAGE WITH NO KINDS. It cannot be written generically — the macro
 * name is PASTED from the row, so there is nothing to `#ifndef` against until the row exists. A package
 * that registers nothing must still define an empty `<name>__LANGUAGE_CONTRACT__OBJECTS(X, PKG)`, and that is the one
 * clause of this contract with no mechanism behind it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ══ HOW THE PACKAGES ARE ASSEMBLED, AND WHY THAT IS NOT THIS FILE ════════════════════════════════════
 *
 * This file is the half a package COMPILES AGAINST, and it includes nothing — which is what lets a
 * package's own implementation reach these macros. `manifest.cuh` next door is the half that INCLUDES
 * the packages. They cannot be one file: a file that a package includes, and that includes the package,
 * is a cycle. The second visit is skipped by the include guard, so the file appears to name what it
 * needs while being handed nothing, and it works for exactly as long as the arrangement above it holds.
 *
 * ── TWO PHASES, AND THE REASON IS THE ONE THE FILES INSIDE A PACKAGE ALREADY USE ─────────────────────
 * A call needs a DECLARATION and never a definition. So every package's declarations arrive first and
 * every package's definitions after all of them, which means one package's implementation may call
 * anything another package declares, whatever order the rows sit in. Position carries no meaning there
 * for the same reason it carries none inside a package.
 *
 * ── THE ASSEMBLY IS GENERATED — `scripts/src_assemble.sh` ───────────────────────────────────────────
 * It is a pure function of the roster above: one include per package per phase, in row order — the
 * program's `manifest.cuh` and a silicon family's `manifest__gpu.cuh`. So it is DERIVED rather than
 * written, and adding a package is adding its folder: the same script enrols it into the roster first.
 * ⛳ THE EXPANSION IS DONE BY `cpp`, handing the real preprocessor this real file and putting the `#`
 * back on what comes out. A script that parsed `PACKAGE_LIST` itself would be a SECOND implementation of
 * what the macro means, free to disagree with the compiler about this file's own roster.
 * ⛔ AND THE ASSERTS IT EMITS ARE NOT MADE REDUNDANT BY IT. A generated file that is committed can be
 * STALE — a row added and the script not run — and a package missing from the assembly is otherwise a
 * link failure with no author. ⇒ ★ THE GENERATOR MAKES THE COMMON CASE ONE EDIT; THE ASSERTS MAKE THE
 * FORGOTTEN CASE LOUD. Neither covers the other's half.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#endif /* SILVANN__PACKAGES_MANIFEST__HEADER_CUH */
