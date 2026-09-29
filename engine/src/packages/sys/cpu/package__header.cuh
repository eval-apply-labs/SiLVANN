#ifndef SILVANN__PACKAGES_SYS_CPU_PACKAGE__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_PACKAGE__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap_node__header.cuh"   /* a handler is handed a base, and a base is a node */
#include "../contracts/objects/package.cuh" /* its constants, fault words and layouts */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 *   · `hatch_stands` and `entry_stands` exist because a package could first ask for no
 *     room; until then an empty entry meant "missing", nn's init refused on it, and the no-config suite
 *     boots then refused too — the third instance in two days, after `host_section` and `host_value`.
 *   · The hatch fill once looped inside a `__global__` and could not be tested without a boot.
 * ⛳ RETIREMENT: at release, with the rest.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ STANDING THE PACKAGES UP, AS A PROGRAM ══════════════════════════════════════════════════════════
 *
 * The language and the evaluator are stood up in C, by the boot, because nothing can run before they
 * are. Everything after that is a package's own business, and the order packages want among themselves
 * is not something the boot can know — so it is not the boot that decides it.
 *
 * ⚖⚖ THE ARCHITECT'S DESIGN: *"init therefore takes the init of all the packages and builds it as a
 * lisp procedure and executes it."* One call per package, in a list, evaluated in a machine that is
 * already whole.
 *
 * ⭐⭐ AND THE LIST IS THE READINESS STATE, WHICH IS THE WHOLE OF WHY THIS NEEDS NO OTHER MECHANISM.
 * ⚖ *"we know that the dependent module is there because otherwise it would not compile and we know it
 * is initialized because its init is not in the list."* Two questions that would each have wanted a
 * channel of their own, and neither does:
 *
 *     DOES IT EXIST?        the compiler answers it. An init that names another package's verb will
 *                           not build without that package, so there is no run-time case to handle.
 *     HAS IT RUN?           the LIST answers it. A call that has not happened is still in there; one
 *                           that has is gone. Presence IS the state, so nothing has to publish a flag,
 *                           nothing has to read one, and the two can never disagree.
 *
 * ⇒ ★ A STATE THAT IS ALREADY IMPLIED BY A STRUCTURE DOES NOT NEED TO BE RECORDED BESIDE IT. The
 * alternative — a readiness row per package — would be a second copy of a fact the list already holds,
 * with its own initialisation, its own ordering and its own way of being stale.
 *
 * ── HOW A PACKAGE WAITS, WHICH IS BY NOT WAITING ─────────────────────────────────────────────────────
 * ⚖ *"he can reorganize its arguments, place a copy of its opcode at the end and return null, letting
 * the next opcode be the new head."*
 * An init that finds a dependency still pending does not block and does not retry in place. It appends a
 * copy of ITS OWN call to the tail of the list and answers nothing; the driver drops the head either
 * way, so the next call becomes the head and runs. The deferred package comes round again once the list
 * has turned over, by which time whatever it was waiting for has either run or been deferred past it.
 * ⛔ NOTHING BLOCKS, AND THAT IS THE PROPERTY WORTH KEEPING. A wait inside an init would be a wait
 * inside a launch, and a launch that waits for something that never happens is this tree's opening
 * warning: a process that hangs with no suspect. A deferral is a rewrite of a list and cannot hang.
 * ⚠ WHAT IT CAN DO IS NOT TERMINATE, WHICH IS A DIFFERENT AND SMALLER PROBLEM. Two packages each
 * deferring behind the other rewrite the list forever. ⚖ THE PROGRESS DETECTOR IS DEFERRED by ruling —
 * a full pass that completes no init is the condition, and it is worth building when there are enough
 * packages for a cycle to exist. What stands here instead is a plain BOUND: the driver gives up after a
 * fixed number of turns and refuses, naming what was still pending. It is not the detector and does not
 * pretend to be; it is the floor that keeps a mistake here from being a hang.
 *
 * ── WHAT AN INIT IS HANDED ───────────────────────────────────────────────────────────────────────────
 * The list, as its one argument. A handler is given its own form and nothing else — it cannot look up at
 * what encloses it — so the thing it has to read has to arrive the ordinary way.
 * ⛳ AND THE FORM IS BUILT PER TURN RATHER THAN STORED IN THE LIST, which is what keeps the reference an
 * ordinary one. A cell naming the list, held INSIDE that list, would be a list that holds itself: a
 * count that can never reach zero and a leak with no event, which is the failure this tree has already
 * had to design a whole structure around. The driver builds `(init <list>)` for the call it is about to
 * make and lets it go afterwards, so nothing ever points at its own container.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* One row of this package's own state, or a null value if the row, the entry or the hatch is not there.
 * A package reads its own by knowing its id, which is the whole of what the hatch is for.
 * ⛔ IT DOES NOT RETAIN — ▶ the perennial premise in `contracts/objects/package.cuh`. A caller that
 * stores the answer somewhere that
 * outlives the machine's stand-up is outside that premise and wants `get`. */
static __device__ inline sys__heap_node sys__package__own_row(uint64_t package_id, uint64_t row);

/* Write one of this package's OWN rows. ⛔ ROW 0 IS REFUSED HERE and not merely discouraged: the room is
 * the boot's to state, and a package that could overwrite it could tell every later reader — including
 * its own teardown — that it holds bytes nobody took. One writer per fact is the whole of the rule that
 * the free mask will be held to as well. */
static __device__ inline bool sys__package__own_row_set(uint64_t package_id, uint64_t row,
                                                        const sys__heap_node* value);

/* This package's own slice, or a null value if it asked for none. ⛳ IT IS `own_row` OF ROW 0, NAMED —
 * kept as its own word because every reader of it asks for the room and not for "row zero". */
static __device__ inline sys__heap_node sys__package__own_room(uint64_t package_id);

/* ⭐⭐ WHETHER THE HATCH ITSELF IS UP — A DIFFERENT QUESTION FROM WHETHER THIS PACKAGE HAS AN ENTRY IN
 * IT. `own_room` answers `nothing` to both: the table is not there, or the table is there and nobody
 * wrote this row. ⛔ AND THE SECOND IS ORDINARY: a package can legitimately ask for no room (`nn`'s span
 * is the sum of its config terms, and a config naming no nn key sums to nothing), so an empty entry
 * usually means *"nobody asked"*, not *"it went missing"* — and a boot with no config at all, which is
 * how most of the suite boots, must not read it as a failure. This is the question that tells them apart.
 * ⇒ ★★ A NEW LEGITIMATE STATE MAKES AN OLD PREDICATE AMBIGUOUS WITHOUT A LINE OF IT CHANGING. ⛳ The
 * question to ask on adding a state is which existing answers just became ambiguous. */
static __device__ inline bool sys__package__hatch_stands(void);

/* Stand the hatch up and fill it. ⛔ `zzengine_` because the boot is the only caller there can be: the
 * array has to exist before the first init runs and the slices are the engine's to hand out, so a second
 * caller would be a package re-publishing the table it is a tenant of. */
static __device__ inline bool sys__package__zzengine_publish_hatch(uint64_t entries);

/* Open one package's sub-array, `own_rows` of its own plus the room's. ⛳ CALLED FOR A PACKAGE THAT HAS
 * SOMETHING TO PUT IN ONE AND NOT FOR EVERY PACKAGE — a package with no rows and no room keeps its null
 * entry, which is the hole the ruling asked for and which `own_row` already answers correctly. */
static __device__ inline bool sys__package__zzengine_open_entry(uint64_t package_id);

/* How many rows of its own a package declared. ⛳ IT IS HERE SO THAT NOBODY OUTSIDE STATES A LENGTH —
 * `open_entry` looks its own up, and nothing else needs to know the number at all. */
static __device__ inline uint64_t sys__package__zzengine_own_rows(uint64_t package_id);

/* ⭐⭐ WHETHER THIS PACKAGE HAS AN ENTRY AT ALL — AND IT IS A THIRD QUESTION, NOT A REPHRASING OF THE
 * OTHER TWO. The hole the ruling asked for is a state of its own, so `own_row` answers `nothing` to THREE
 * different facts:
 *     the hatch is not up            ⇒ `hatch_stands` is false
 *     the hatch is up, no entry      ⇒ this package keeps nothing of its own and none was opened
 *     an entry is open, row unwritten ⇒ the ordinary state of a row before somebody fills it
 * ⇒ ★★ A NEW LEGITIMATE STATE IS WHAT MAKES AN OLD PREDICATE AMBIGUOUS — the same lesson this file
 * records above about `hatch_stands`, for a second legitimate emptiness.
 * ⛳ AND IT IS WHAT MAKES THE HOLE OBSERVABLE: without this, a boot that opened
 * an entry for every package and a boot that opened none would answer identically to every reader, and
 * the ruling would be a sentence nothing could check. */
static __device__ inline bool sys__package__entry_stands(uint64_t package_id);

/* ⭐ FILL THE HATCH FROM WHAT THE HOST TOOK — `SYS__PACKAGE__ROOM_SLICE` words per package, in roster
 * order. ⛳ IT LIVES HERE AND NOT IN THE ENGINE because the rule it applies is the HATCH's: who gets an
 * entry, and how long it is. The engine has the addresses and a runner to write them with, and that is
 * all it contributes — so the decision is testable without a boot. */
static __device__ inline bool sys__package__zzengine_fill_hatch(const uint64_t* slices);
static __device__ inline bool sys__package__zzengine_hand_room(uint64_t package_id, uint64_t at,
                                                               uint64_t bytes);
static __device__ inline bool sys__package__zzengine_hand_ram(uint64_t package_id, uint64_t ram_at,
                                                              uint64_t ram_card_at, uint64_t ram_bytes);

/* The list an init was handed, or zero with `why` set to the word that says what was wrong instead.
 * ⛳ IT ANSWERS RATHER THAN REFUSING, and the caller refuses. A helper that raised on its caller's behalf
 * would put the refusal out of sight of the `return` that follows it — so a reader, and the check that
 * reads after them, would both meet a verb walking away from its form with nothing written into it. */
static __device__ inline uint64_t sys__package__list_of(uint64_t form, uint64_t* why);

/* Refuse, and answer the refusal — which is what every package's init does with the word above. It is
 * here, and public, because a package that is not `sys` has no other way to refuse: the opcodes' own
 * refusal is package-internal, and reaching for it from outside is a trespass the C subset gate catches. */
static __device__ inline void sys__package__refuses(uint64_t form, uint64_t why);

/* Whether a call to `verb` is still somewhere in `list` — i.e. whether that package has yet to run.
 * ⛳ IT ASKS ABOUT A VERB AND NOT ABOUT A PACKAGE, because a verb id is a compile-time constant a caller
 * can name and a package is not a thing the language has a value for. */
static __device__ inline bool sys__package__waiting_for(uint64_t list, uint64_t verb);

/* Put `verb` back at the tail, so the caller runs again after everything currently ahead of it. Answers
 * false only if there was no room, which the caller must turn into a refusal rather than a silent skip —
 * an init that meant to defer and did not is an init that never ran. */
static __device__ inline bool sys__package__come_back_later(uint64_t list, uint64_t verb);

#endif /* SILVANN__PACKAGES_SYS_CPU_PACKAGE__HEADER_CUH */
