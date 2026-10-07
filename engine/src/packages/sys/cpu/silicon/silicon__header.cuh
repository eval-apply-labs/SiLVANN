#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__HEADER_CUH

/* Nothing in this package has to come first: this file names none of it. */
/* ⛳ `size_t`, AND IT IS THE ONLY THING THIS FILE TAKES FROM OUTSIDE ITSELF. The host verbs at the foot
 * of this file are declared in the width the vendor runtimes use, and a declaration that does not match
 * its definition is the one drift this seam's conformance rule exists to catch — so the type has to be
 * named here, not arrive by luck through whatever the harness had already included: an include that
 * arrives through somebody else is a dependency nobody wrote down, and it is invisible right up until
 * the somebody else has a reason to leave. `<stddef.h>` is freestanding, so a target with no
 * hosted library still has it. */
#include <stddef.h>
#include "../../contracts/abi/silicon_family.cuh"   /* what a family id is, which every door verb names first */

/* ══ THE SILICON SEAM — WHAT A BACKEND OWES ══════════════════════════════════════════════════════════
 *
 * The primitives below. Everything else in `sys` is written in the C subset and runs anywhere; these
 * are the places where the hardware is not the same, and a backend is a file that answers every one.
 *
 * ⭐⭐ THIS FILE IS ADDRESSED TO SOMEBODY PORTING, NOT SOMEBODY CALLING, WHICH IS WHY ITS CONTRACT RUNS
 * THE OTHER WAY ROUND FROM THE REST OF THIS PACKAGE. Elsewhere the three lists say what a CALLER owes and
 * what the code promises back. Here the implementer is the reader: the list below is what YOUR BACKEND
 * owes, what the package promises about how it will be CALLED, and — the one that saves the most work —
 * what it does NOT require, so a port is not built stronger than it needs to be.
 *
 * ⛳ AND THE CONTRACTS ARE NARROWER THAN THE HARDWARE'S EVERY TIME. That is deliberate: what is promised
 * here is what the callers in this package actually rely on, so a target that offers less than its
 * datasheet suggests may still be able to answer, and one that offers more owes nothing extra.
 *
 * ⛔⛔ AND THE SEAM HAS TWO SIDES, WHICH IS THE FIRST THING TO KNOW BECAUSE THEIR PROMISES ARE
 * OPPOSITE. The EVALUATOR verbs — the `__device__`-marked ones — run on the evaluator's runner threads,
 * inside its hot paths, and may not block or allocate. The DOOR verbs are ordinary host functions that
 * reach a card through its silicon family, and their entire job is allocating and copying — blocking is
 * what several of them are FOR. A rule stated for one side is wrong for the other, so every one below
 * says which side it binds, and the door verbs all wear `sys__gpu__` in their names so a call site says
 * which side it is on without anybody having to look the verb up.
 * ⚠ `publish_to_host` IS AN EVALUATOR VERB DESPITE ITS NAME. It says where what it publishes is going, not
 * who calls it — it is the one place in this file where "host" is a destination rather than a caller.
 *
 * ── WHAT A BACKEND MUST NOT DO — THE EVALUATOR SIDE ─────────────────────────────────────────────────
 *   never raise      no primitive here reports through `sys__fault__raise`. A refusal is a RETURN
 *                    VALUE — null from `alloc`, the found word from `cas_u32` — because only the caller
 *                    knows whether it can carry on without what it asked for. A backend that raises
 *                    decides that for it.
 *   never block      nothing here may wait on another thread, take a lock the caller cannot see, or spin
 *                    unbounded. `wait_cycles` waits ON PURPOSE and for a bounded length; every other one
 *                    returns in finite time no matter what any other block is doing.
 *   never allocate   except `alloc`. In particular `cas_u32` and `wait_cycles` run on paths that have no
 *                    memory to spare and are called from inside allocation itself.
 *
 * ── WHAT THE PACKAGE PROMISES ABOUT HOW IT CALLS THEM — THE EVALUATOR SIDE ──────────────────────────
 *   one thread       every call site is single-threaded per block, or is a compare-and-swap. A backend
 *                    needs no per-lane election and none is done for it — see the note in
 *                    `heap_object__impl.cuh` about lane-level contention being the caller's debt.
 *   evaluator only   every one is called from the evaluator's runner threads.
 *   rarely, mostly   `alloc` and `free` run when infrastructure is built, not per value. `cas_u32` is the
 *                    hot one; `wait_cycles` is what an idle block runs once per turn — mostly after a
 *                    plain READ found nothing to do, sometimes after a swap came up short.
 *
 * ── AND THE SAME TWO LISTS FOR THE DOOR SIDE, WHICH READ NOTHING ALIKE ──────────────────────────────
 *   never raise      the one rule both sides share, and for the same reason. A `bool` back, and the
 *                    family's own vocabulary of failure stops here. See the host contract below.
 *   blocking is fine several of them wait on the device by definition — `sys__gpu__compute_completed`
 *                    is nothing else.
 *   allocating is    `sys__gpu__memory_allocate`, `sys__gpu__memory_allocate_host_ram_mapped` and
 *   the job          `sys__gpu__side_open` all acquire; that is what they are.
 *   one thread       the package makes no promise that two threads may call these at once. Whatever
 *                    stands the engine up calls them, and that is one thread today.
 *   host only        ordinary host functions, and none may launch anything. A seam
 *                    that launched would be deciding the engine's shape rather than serving it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE COMPARE-AND-SWAP ────────────────────────────────────────────────────────────────────────────
 *
 * Compare-and-swap exists on every target this engine could run on, and none of them spell it the same
 * way or promise quite the same thing. HIP and CUDA offer it as a function that answers with the word it
 * found. Vulkan offers an instruction that takes a scope and a set of memory semantics as operands. C11
 * and the ARM targets that go through it take the expected value BY POINTER, overwrite it on failure, and
 * report success separately — and offer a second form that is allowed to fail for no reason at all.
 *
 * ── THE CONTRACT EVERY BACKEND OWES, AND IT IS NARROWER THAN THE HARDWARE'S ──────────────────────────
 *
 *     read the word · if it equals `expected`, store `desired` · return what was read, either way
 *
 * The return value is the whole interface: a caller learns whether the store was its own by comparing
 * what came back against what it planned against, and needs no second output to do it.
 *
 * ⛔ THAT MAKES ONE PROPERTY LOAD-BEARING RATHER THAN INCIDENTAL: **the swap must not fail when the word
 * matched.** A backend built on the weak form of compare-exchange would sometimes leave the word untouched
 * and still hand back a value equal to `expected` — and every caller in this tree would read that as "the
 * store was mine" and carry on having changed nothing. The counter it was decrementing would keep its old
 * value while the caller believed it had taken it down, which is a lost update wearing a success's
 * clothes. A backend that has only the weak form owes a loop here, not a rename.
 *
 * The width is in the name because the widths available differ per target, and a caller that needs 64 bits
 * should fail to compile on a backend without it rather than silently get 32.
 *
 * Scope is device-wide and ordering is relaxed, which is what the callers want: the allocator's claim and
 * the reference counts are decided BY the compare-and-swap itself, not by what they order around it. A
 * caller that needs a fence asks for one — the barrier does, and says so where it does it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── WAITING, ON PURPOSE AND FOR A KNOWN LENGTH ──────────────────────────────────────────────────────
 * Burn approximately `n` cycles. It is here rather than written inline because "how do I wait" is one of
 * the things that genuinely differs per target: this one has a sleep instruction whose argument is in
 * units of 64 cycles, another has a nanosecond sleep, and a target with neither owes a loop the compiler
 * will not delete.
 *
 * ⛔ APPROXIMATELY IS THE HONEST WORD AND IT IS ENOUGH — BUT APPROXIMATELY IS NOT ARBITRARILY, AND THE
 * CALLERS WANT THIS LENGTH FOR THREE DIFFERENT THINGS. One of them, `sys__heap_object__lock`, adds the
 * block id so that two waiters do not resume together, and there only the DIFFERENCE matters. The rest
 * want the ABSOLUTE length. `sys__computing_base__begin` and the parked block's idle arm wait because an
 * idle block polling a word at full rate pays the traffic for having nothing to do — that is the whole of
 * why the wait is there, and it is deliberately the same delay for every block, since block N polls base
 * N and has no rival to fall out of phase with. And three sites multiply this length by a try count to
 * make a TIMEOUT, so the number decides an ANSWER: `ENG__ABI__DISPATCH_TIMED_OUT`,
 * `SYS__CAROUSEL__FAULT_STALE`, and the result poll that gives up after `SYS__COMPUTING_BASE__RESULT_TRIES`.
 * ⇒ ★ A BACKEND THAT MAKES THIS CHEAP BREAKS NO CORRECTNESS ARGUMENT; it restores full-rate idle polling
 * and turns a legitimately slow producer into a stale answer. Nobody here needs an exact interval, and
 * everybody needs roughly the length they asked for.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── ALLOCATION ──────────────────────────────────────────────────────────────────────────────────────
 *
 * Room from the PLATFORM, not from this engine's heap. Two verbs, and they exist so that a structure can
 * be "a C object beyond the heap" and still be made at run time rather than declared at compile time.
 *
 * ── THE CONTRACT EVERY BACKEND OWES ─────────────────────────────────────────────────────────────────
 *
 *     alloc   `bytes` of room aligned for any type, or NULL. NEVER a partial answer, never a smaller
 *             block than asked for, and never a raise — a caller that can carry on without the room is
 *             the one who knows it, so the refusal is a value and not a fault.
 *     free    the room from one `alloc`, once. Freeing NULL is a no-op, which is what lets a caller
 *             clean up without first asking whether there is anything to clean.
 *
 * ⛳ ON THE HOST BACKEND IT IS THE C LIBRARY'S `malloc` (`cpu/silicon/memory.cuh`), with no limit of its
 * own. ⛔ A BACKEND WHOSE ALLOCATOR HAS A LIMIT SET SOMEWHERE ELSE — a device arena sized from outside —
 * owes that limit at boot, and says so beside its definition: a refusal there means the limit, not the
 * room. ⇒ ★ A SEAM CAN ONLY PROMISE WHAT ITS BACKEND PROMISES.
 *
 * ⚠ THE COST IS `REASONED`. Nothing has measured what `alloc` does under contention between runners,
 * and an allocator that serialises its callers would answer every one of them correctly and slowly. What
 * would falsify THAT: a boot whose time grows with the number of blocks asking.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── THE PRIMITIVES, DECLARED───────────────────────────────────────────────────────────────────────
 * Each is DEFINED by exactly one backend, selected by the build. A backend missing one does not link —
 * these are `static inline`, so a declared-but-undefined primitive that anything calls is an error
 * naming it, which is the cheapest possible conformance check and needs no list to maintain.
 *
 * ⛔⛔ AND IN THE HARNESS IT IS HALF A CHECK, WHICH IS THE OPPOSITE OF WHAT THE ARRANGEMENT LOOKS LIKE.
 * A stand-in that drifts from the seam does NOT reliably fail to compile. `MEASURED`, by injecting each
 * kind of drift into these declarations and building:
 *
 *     RETURN type drift     C++ harness: caught AT COMPILE — *"ambiguating new declaration"*
 *     PARAMETER type drift  C++ harness: 0 COMPILE ERRORS, caught AT LINK. C++ overloads, so the drifted
 *                           stand-in is a second function and the compiler is content — but every caller
 *                           was compiled against the DECLARED signature, and nothing defines that one.
 *                           `MEASURED`: one stand-in parameter changed from `unsigned int*` to
 *                           `int*`, harness built unmodified as the control (0 errors, binary produced) and
 *                           drifted (0 compile errors, no binary) — *"undefined reference to
 *                           `sys__silicon__add_u32(unsigned int*, unsigned int)'"*.
 *     PARAMETER type drift  C: caught AT COMPILE — *"conflicting types for 'f'"*
 *
 * ⇒ ★ SO BOTH COMPILATIONS CATCH IT AND THEY CATCH IT AT DIFFERENT MOMENTS, which is worth knowing
 * because the two failures read nothing alike: C names the drifted line, C++ names an address in the
 * CALLER and a signature nobody wrote. A reader meeting the second without expecting it goes looking at
 * the wrong file.
 * ⛔⛔ AND THE C++ CATCH HAS A CONDITION THE C ONE DOES NOT: it fires only for a verb SOMETHING CALLS.
 * A drifted verb with no caller links perfectly, because there is no unresolved reference to be left —
 * so the seam can be quietly wrong in exactly the verbs a test does not exercise, which is the shape
 * every hole in this tree has had. ⇒ ★ THAT GAP IS WHY THE CONFORMANCE RULE EXISTS RATHER THAN BEING
 * BELT AND BRACES: it compares every declared verb against every backend's definitions by SIGNATURE,
 * caller or no caller, and it is the only check here that does not need somebody to have called the
 * thing first.
 * ⛳ THAT IS STILL WORTH HAVING and it is worth knowing which half: this package has been bitten three
 * times by a stand-in that drifted from what it stood for, each time invisibly because the suite was green
 * against the stand-in. Half a check that names its own gap beats a whole one nobody proved.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── WHICH BLOCK IS ASKING ───────────────────────────────────────────────────────────────────────────
 *
 * The index of the calling block within the launch. It is a HARDWARE question — the answer is read out of
 * the execution state and there is nowhere else it could come from — which is why it is behind this seam
 * and not written where it is wanted.
 *
 * ⛔ AND IT IS THE ONE PLACE A TARGET'S OWN SPELLING WOULD OTHERWISE LEAK INTO THE PACKAGE. Written where
 * it is wanted, it would be a vendor intrinsic in the allocator's header, and porting would mean finding
 * it there rather than here with everything else that differs.
 *
 * ── THE CONTRACT EVERY BACKEND OWES ─────────────────────────────────────────────────────────────────
 * It is UNIFORM ACROSS A WAVEFRONT — every lane of a block answers the same — which is what lets a caller
 * treat the value as a scalar and lets the compiler hoist a lookup keyed on it out of a loop.
 * It is STABLE for the life of the block: a block does not become another block.
 * It is DENSE from zero, because callers index arrays with it, and one of them is the table of computing
 * bases with a slot per block.
 * ⛳ OFF THE DEVICE A HOST **HARNESS** ANSWERS ZERO, BECAUSE IT RUNS ONE BLOCK — but a host EVALUATOR
 * does not. `cpu/silicon/block.cuh` answers a `__thread` runner id below
 * `SYS__SILICON__HOST_MAX_RUNNERS`, and its own header says why: an id that is always 0 would make every
 * thread claim the SAME computing base, turning four GPUs into one, silently.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── WHERE A VALUE LIVES WHEN IT BELONGS TO THE BLOCK ────────────────────────────────────────────────
 * A thread's frame is private and a block's storage is not, and which one a value wants is a property
 * of what the package is running inside rather than of the value. On a device the block's storage is
 * the shared segment; where there is one block of one thread, an ordinary variable serves — so long as
 * it OUTLIVES THE CALL, which is the whole of why this carries `static`.
 *
 * ⛔ THE LIFETIME IS THE LOAD-BEARING HALF, NOT THE SHARING. A declaration like this belongs inside a
 * function, because that is the only place the shared segment may be named. Where it is READ AND
 * WRITTEN INSIDE THAT SAME FUNCTION a plain local would serve. What does not survive is a POINTER TO IT
 * ESCAPING: on a host standing in for the segment a plain local lives in the caller's frame, so an
 * accessor handing one out hands out a pointer to a frame that has gone. `static` is what makes one
 * spelling mean block lifetime on either side, and it is what such a host must supply too.
 *
 * ⭐ IT IS A STORAGE CLASS AND SO IT IS A MACRO, not one of the functions below. There is nothing
 * to call: the declaration is the whole of it.
 *
 * ⛳ AND THE SHIPPING DEFAULT IS DECLARED OUT HERE while only the override is supplied by whoever is
 * hosting the package — a deleted override must leave a working value behind, and this is it.
 *
 * ⛔⛔ AND THE OVERRIDE IS NOT OPTIONAL ON A HOST EVALUATOR — IT IS THE LINE THAT DECIDES WHETHER FOUR
 * CARDS RUN AS FOUR. Everything above reasons from *"where there is one block of one thread"*, and that
 * premise is false for the shipping host build: ⚖ *"one thread per gpu and one for the orchestrator at
 * id 0."* The host analogue of "per BLOCK" is "per RUNNER THREAD", which is `__thread` and is NOT
 * `static` — so `cpu/silicon/environment.cuh` defines it `static __thread`.
 * ⛔ PLAIN `static` COMPILES, PASSES EVERY SINGLE-RUNNER TEST, AND IS THE BUG: at one runner the two
 * spellings are indistinguishable, and at eight they differ by every tenant of this macro sharing ONE
 * copy. `test/src_host_shim.cuh` defines exactly that — correctly, because a harness runs one block —
 * which is why the shim is not promotable and why `test/src_host_evaluator_smoke.cpp` exists to exercise
 * the real one.
 * ⇒ ★★ THERE IS NO FAILURE TO OBSERVE IF THIS IS WRONG — the claim protocol serialises the runners
 * perfectly and silently, and the only symptom is a number that is never as large as it should be.
 * ⛳ TENANTS, so a reader knows the blast radius: `bindings__impl` · `engine/eval.cuh`'s loop-carried
 * state · `heap__impl` ×2. The evaluator's own hot state, all of it. */
#ifndef SYS__SILICON__BLOCK_LOCAL
#define SYS__SILICON__BLOCK_LOCAL static __shared__
#endif

static __device__ inline unsigned int sys__silicon__block_id(void);

/* ── HOW MANY BLOCKS ARE RUNNING ─────────────────────────────────────────────────────────────────────
 *
 * The number of blocks in this launch. ⛔ IT IS NOT A MAXIMUM AND NOT A PROPERTY OF THE CARD — it is
 * what the launch ASKED FOR, so a caller that deliberately runs a narrow grid gets the narrow number and
 * not the width the hardware could have managed.
 *
 * ⛔⛔ AND IT IS NOT A PROMISE THAT EVERY BLOCK IN IT IS DOING ANYTHING. `eng__boot` takes the width it is
 * handed verbatim and starts one runner thread per block, and nothing in this tree checks that a runner
 * it SCHEDULES onto is one that was started.
 * ⇒ ★ NOTHING FAILS WHEN IT IS WRONG, WHICH IS THE HAZARD. `src` has no rendezvous, so a block that never
 * gets work simply never gets work. What does strand is a caller that schedules onto a block number past
 * the launch: it polls a word nobody will ever write, with nothing in the runtime to notice.
 * ⇒ ⛳ SO FITTING THE WORK TO THE LAUNCH IS THE CALLER'S DEBT, not something this count enforces or a
 * backend can check. What the number promises is the width of the ASK.
 *
 * ── THE CONTRACT EVERY BACKEND OWES ─────────────────────────────────────────────────────────────────
 * Same for every block in the launch and for the whole life of it.
 * At least one, because something is asking.
 * `block_id` is DENSE below it: the ids are `0 .. count-1`, which is what lets a caller size a table
 * with this and index it with that.
 * ⛳ OFF THE DEVICE A HOST HARNESS ANSWERS ONE, because it runs one block. ⛔ A host EVALUATOR answers
 * however many runner threads it was given — ▶ `block_id` above, same cause.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline unsigned int sys__silicon__block_count(void);

/* ⭐ WHICH WORKER THE CALLER IS, AND HOW MANY THE MACHINE HAS — a worker is one device of one family, and block
 * `w` runs as worker `w`; a block past the last runs as worker 0. ▶ `block.cuh`. */
static __device__ inline unsigned int sys__silicon__worker(void);
static __device__ inline unsigned int sys__silicon__worker_count(void);

static __device__ inline unsigned int sys__silicon__cas_u32(unsigned int* address,
                                                            unsigned int expected,
                                                            unsigned int desired);
/* Add to a word and answer what was there before. ⭐ IT IS HERE BECAUSE `cas_u32` ALONE MAKES EVERY
 * COUNTER A RETRY LOOP: a plain increment written as `for(;;) { cas(slot, cur, cur+1); }` is a structure
 * the compiler must keep whole, and this tree has measured repeatedly that the structure is what costs
 * and the atomic is not. The hardware has a single instruction for this; the seam simply had no word for
 * it, so the package wrote the loop instead. */
static __device__ inline unsigned int sys__silicon__add_u32(unsigned int* address, unsigned int addend);

/* ── READING A WORD SOMEBODY OUTSIDE THIS BLOCK IS WRITING, WITHOUT WRITING IT BACK ──────────────────
 *
 * ⭐⭐ IT EXISTS BECAUSE THE TWO WORDS ABOVE ARE THE ONLY UNCACHEABLE READS THE SEAM HAD, AND BOTH OF
 *   THEM WRITE. Adding zero and comparing-and-swapping to the same value both read the word — and both
 *   put it back. That is harmless between blocks, which share a cache and so put back what is really
 *   there. It is NOT harmless against a writer outside the device: a poll that stores the value it just
 *   read will, sooner or later, store a value that was already superseded, and the store that is lost is
 *   the one the other side was counting on.
 *
 * ⛔⛔ THIS IS `MEASURED` AND IT COST THE FIRST BUILD OF HOST-SIDE SCHEDULING. A block waiting for work
 *   compare-and-swapped its own status word about a million times a second; the host wrote SCHEDULED
 *   into it over the same channel that had just written the two operands beside it, and only the
 *   operands survived. Placing the identical work BEFORE the launch and letting the same block find it
 *   ran correctly and answered — so the loop, the fill and the reading were all right, and what was
 *   wrong was that the waiter wrote to what it was watching.
 * ⇒ ★★ **A POLL THAT WRITES IS A POLL THAT OVERWRITES**, and it is invisible until a second writer
 *   exists, because until then it only ever puts back the value it found.
 *
 * ⛳ IT IS ALSO STRICTLY CHEAPER. A waiting block now reads rather than performing a read-modify-write on
 *   a line every other waiting block is also holding.
 * ⛔ AND IT IS NOT AN ORDERING PRIMITIVE. It says nothing about what else is visible; `publish` and
 *   `publish_to_host` are what say that. This only promises the word itself is fetched. */
static __device__ inline unsigned int sys__silicon__read_u32(const unsigned int* address);

/* ── MAKING WHAT YOU WROTE VISIBLE TO SOMEBODY ELSE ──────────────────────────────────────────────────
 *
 * Everything this block has written lands before anything it writes after this does. It moves no data and
 * answers nothing; what it buys is an ORDER, and the order is only worth having where a second block is
 * reading.
 *
 * ⭐⭐ WHY IT IS NEEDED AT ALL, WHICH IS NOT OBVIOUS WHEN EVERYTHING RUNS IN ONE BLOCK. A block that fills
 * a structure and then sets a flag saying it is filled has written two things, and nothing makes a reader
 * see them in that order — the flag can land first, and the reader then works on a structure that is still
 * whatever the last tenant left. One block cannot catch itself doing this, because it is its own reader
 * and sees its own writes in order by construction. So this is a primitive with no use until the moment
 * there are two, and at that moment every hand-off needs it.
 *
 * ⛔ IT IS NOT A LOCK AND IT IS NOT A BARRIER. Nothing waits, nobody rendezvouses, and no block learns
 * anything about where another has got to. That is what makes it safe to use in a spin: a barrier a block
 * fails to arrive at strands the grid, and this cannot be arrived at or missed.
 *
 * ⛳ AND IT IS DEVICE-WIDE AND NOT HOST-WIDE, which is a real distinction and the narrower of the two on
 * purpose. What it orders is what another BLOCK reads, and that is almost all of the traffic here — so
 * widening it would tax every hand-off between blocks to pay for the one case below. */
static __device__ inline void         sys__silicon__publish(void);

/* ── AND THE SAME THING, WHERE THE SOMEBODY ELSE IS NOT ON THIS DEVICE ───────────────────────────────
 *
 * The one above orders what another block reads. This one orders what the HOST reads, while the launch
 * that wrote it is still running — which is a wider claim and costs more, so the two are separate words
 * and a caller says which reader it means.
 *
 * ⭐⭐ WHY A BLOCK CAN PUBLISH TO ANOTHER BLOCK AND NOT, BY THE SAME ACT, TO THE HOST. The two readers
 *   arrive by different routes: another block reads through the memory the device shares with it, and
 *   the host reads by having a copy engine fetch the words out. A write that has reached the point where
 *   every block would see it has not necessarily reached the point where something outside the device
 *   would, and nothing in the narrower word promises that it has.
 *
 * ⛔⛔ AND WHETHER THE NARROWER ONE WOULD DO ON THIS CARD IS **UNKNOWN**, WHICH IS WHY THIS EXISTS. It
 *   was measured and the measurement came back empty-handed: with the flag and the words it vouches for
 *   sampled in ONE transfer, 400 hand-offs each, the device-scope arm tore nothing — and so did an arm
 *   with NO FENCE AT ALL. A control that cannot fail cannot acquit, so both clean runs say only that the
 *   instrument is blind to the difference. ⇒ ★★ **A CLEAN RUN IS EVIDENCE ONLY IF A DIRTY ONE WAS
 *   POSSIBLE.**
 *
 * ⚖ SO THE CHOICE IS MADE ON THE PRICE AND NOT ON THE PHYSICS, and the price is `MEASURED` **nothing**:
 *   0.106 ms against 0.107 ms per hand-off over 400 rounds each, inside the drift. Correctness resting on
 *   a premise nobody has checked, bought for free, is not a trade worth thinking about twice.
 *   ▶ `measurements/2026-09-14_S101_resident_launch_host_seam.md` ③ */
static __device__ inline void         sys__silicon__publish_to_host(void);

static __device__ inline void         sys__silicon__wait_cycles(unsigned int n);

static __device__ inline void*        sys__silicon__alloc(unsigned int bytes);
static __device__ inline void         sys__silicon__free(void* room);

/* ══ THE HOST SIDE ══════════════════════════════════════════════════════════════════════════════════
 *
 * Everything above runs on the evaluator's runner threads. These twelve are the doors: what the engine
 * and a verb call to get memory onto a device, get answers back off it, and wait for it to go quiet.
 * ⭐ EACH NAMES THE FAMILY IT IS FOR, as its first argument: the program holds a table per family, and a
 * caller says which silicon it means rather than inheriting one from the process. ▶ `silicon_family.cuh`.
 * They are a seam for the same reason the evaluator verbs are — every one of them is a place where two
 * families spell the same idea differently — and they are HERE, in the package, rather than beside the
 * code that calls them, because a caller in one tier cannot be reached from another and this package is
 * the only thing every tier may name.
 *
 * ⛳ THE RETURN IS A YES OR A NO. Whichever family is underneath has its own vocabulary of failure, and
 * carrying it upward would put that vocabulary in the caller's types. What a caller can do about a
 * failed allocation does not depend on which code came back.
 *
 * ⛔ `sys__gpu__memory_allocate` AND `alloc` ARE DIFFERENT POOLS AND THE WORDS MUST NOT MERGE. `alloc`
 * is scratch room for the evaluator itself, in the evaluator's own memory. `sys__gpu__memory_allocate` is
 * asking a family for device memory, which is the ordinary allocation everybody means. A backend that
 * served one from the other would be answering a different question.
 *
 * ── AND A SECOND PATH TO THE SAME MEMORY, FOR READING IT WHILE SOMETHING IS USING IT ────────────────
 * ⭐⭐ `memory_write`, `memory_read`, `memory_zerofill` AND `compute_completed` QUEUE BEHIND THE WORK,
 *   WHICH IS EXACTLY RIGHT UNTIL THE WORK DOES NOT END. Each is issued to the stream every kernel is
 *   launched on,
 *   so a copy is ordered after the kernel before it and a caller reads what that kernel left. A launch
 *   that OUTLIVES the call turns that same property into a deadlock: the copy waits for a kernel that is
 *   waiting to be told what to do.
 * ⭐ AND THAT IS WHAT MADE IT LOOK LIKE THE HOST COULD NOT REACH A RUNNING LAUNCH AT ALL. It can. It has
 *   always been able to — the copy engine is a separate piece of hardware and fetches from memory the
 *   kernels are using. What it cannot do is get a turn on a stream something else is occupying.
 *   `MEASURED`: a copy issued on a channel of its own returns in ~106 µs while a kernel sits spinning on
 *   the ordinary one, over 1,200 hand-offs, with the kernel's own counter proving it was still running.
 *   ⇒ ★★ **A MECHANISM THAT IS BLOCKED BY SOMETHING ELSE LOOKS EXACTLY LIKE A MECHANISM THAT IS ABSENT.**
 *   ▶ `measurements/2026-09-14_S101_resident_launch_host_seam.md` ②
 * ⛔⛔ AND THE ROOM ON THIS SIDE HAS TO BE PINNED OR THE POINT IS LOST. A copy that does not wait for the
 *   device may still be staged through a buffer of the family's own, and the family waits for THAT —
 *   so an unpinned destination gives back a call that blocks for a reason that has nothing to do with the
 *   hardware, and looks from here exactly like the deadlock this exists to avoid.
 * ⛳ THE CHANNEL IS A `void*` LIKE EVERY OTHER HANDLE HERE. Which family is underneath is the seam's
 *   business and not the caller's, and naming its type would put that vocabulary in the caller.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
static inline bool sys__gpu__memory_allocate(sys__silicon_family__id family, void** at, size_t bytes);
static inline void sys__gpu__memory_free(sys__silicon_family__id family, void* at);
static inline bool sys__gpu__memory_zerofill(sys__silicon_family__id family, void* at, size_t bytes);
static inline bool sys__gpu__memory_read(sys__silicon_family__id family, void* to, const void* from, size_t bytes);
static inline bool sys__gpu__memory_write(sys__silicon_family__id family, void* to, const void* from, size_t bytes);
static inline bool sys__gpu__compute_completed(sys__silicon_family__id family);
static inline bool sys__gpu__side_open(sys__silicon_family__id family, void** channel);
static inline void sys__gpu__side_close(sys__silicon_family__id family, void* channel);
static inline bool sys__gpu__side_memory_read(sys__silicon_family__id family, void* to, const void* from, size_t bytes,
                                                void* channel);
static inline bool sys__gpu__side_memory_write(sys__silicon_family__id family, void* to, const void* from, size_t bytes,
                                                void* channel);
static inline void sys__gpu__memory_free_host_ram(sys__silicon_family__id family, void* at);
static inline bool sys__gpu__memory_allocate_host_ram_mapped(sys__silicon_family__id family, void** at, void** card_at, size_t bytes);
static inline bool sys__gpu__memory_register_host(sys__silicon_family__id family, void* at, size_t bytes, void** card_at);
static inline void sys__gpu__memory_unregister_host(sys__silicon_family__id family, void* at);
static inline bool sys__gpu__file_open(sys__silicon_family__id family, const char* path, uint64_t* handle);
static inline void sys__gpu__file_close(sys__silicon_family__id family, uint64_t handle);
static inline bool sys__gpu__file_read(sys__silicon_family__id family, uint64_t handle, uint64_t offset, uint64_t bytes, void* to);
static inline bool sys__gpu__memory_map_file(sys__silicon_family__id family, void* at, size_t bytes, uint64_t handle, uint64_t offset);
static inline bool sys__gpu__memory_prefetch(sys__silicon_family__id family, const void* at, size_t bytes);

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON__HEADER_CUH */
