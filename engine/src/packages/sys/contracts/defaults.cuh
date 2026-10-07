#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_DEFAULTS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_DEFAULTS_CUH

/* Nothing in this package has to come first: this file names none of it. */
/* ══ the numbers you can turn, and what they are when nobody turns them ══════════════════════════════
 *
 * Values that are a CHOICE rather than a consequence. Everything here could be different without anything
 * else being wrong — which is exactly what makes them worth keeping apart from the code that reads them,
 * because a number sitting beside its one user reads as a property of that code rather than as a dial.
 *
 * ⭐⭐ THIS IS WHERE A NUMBER IS TURNED, NOT WHERE IT IS GUARANTEED. Every entry below is ALSO guarded in
 * the file that uses it, with the same value, so commenting one out here leaves a working program rather
 * than a build error. ⚖ ARCHITECT: *"i can comment out the default in the variables and not set it at
 * compile time and i still would like a functioning program."*
 * ⇒ ★ AN OVERRIDE IS A THING YOU MUST BE ABLE TO DELETE, which is the internal-lever rule one tier up
 * applied to a dial: the arming is optional, the value is not.
 * ⚠ THE PRICE IS TWO COPIES THAT CAN DISAGREE, and nothing checks it — by the time a consumer is read
 * only one of the two definitions still exists, so no assert can compare them. Change one, change both.
 *
 * ⛳ EACH IS `#ifndef`-GUARDED SO A HARNESS OR A BUILD CAN OVERRIDE IT WITHOUT EDITING ANYTHING. The
 * default is what ships; the override is what a geometry test or a measurement uses. That is the same
 * shape as the internal-lever rule one tier up — the shipping value lives outside the marker, the
 * override goes in it — and it is why the selftest can run this package at three geometries.
 *
 * ⚖ ARCHITECT: *"should SYS__HEAP_OBJECT__LOCK_TRIES be an ifdef and be written in variables? which btw
 * should likely be renamed as variables_default."* Both — and the second is why the first was easy to
 * miss. The name says what the file holds, which is one `#ifndef`-guarded DEFAULT per dial. A file
 * named for a CATEGORY invites a reader to ask whether a number belongs in it; one named for a SHAPE
 * lets them ask whether the number HAS that shape, which is answerable by looking.
 *
 * ⛔ WHAT DOES NOT BELONG HERE: a number that is forced. A heap chunk is 512 nodes because a chunk sits at
 * a fixed stride and clearing the low bits of an offset anywhere inside one has to land on its first node,
 * and because that is what makes an in-chunk offset fit the width the arithmetic wants. That is a conclusion
 * with a proof beside it, and moving it here would invite somebody to turn it.
 * ⚠ THE ARRAY GRANULE READS LIKE THE SAME KIND OF NUMBER AND IS NOT, WHICH IS WORTH SAYING BECAUSE THE
 * MISTAKE IS EASY. `SYS__ALLOC__ARRAY_GRANULE` is a CAPACITY — how many cells one link of a chain holds —
 * nothing is rounded to it and no address in this package is ever masked. It is `#ifndef`-guarded in exactly
 * the shape everything below has, it is set by preference rather than derived, and it is split into two
 * dials of its own, `SYS__STACK_CHUNK__ALLOC` and `SYS__LIST_CHUNK__ALLOC`, because a deep binding stack and
 * a list of three-value forms want different numbers.
 * ⇒ ★ SO IT HAS THIS FILE'S SHAPE AND STILL LIVES IN `heap__header.cuh`, AND THAT IS THE SECOND HALF OF THE
 * MEMBERSHIP TEST: a kind's SIZE is stated in that kind's row, where every other kind states its own. Shape
 * says whether a number may be turned; the row says where it is read.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* How long a block waits between one look at a computing base and the next.
 *
 * ⚖ ARCHITECT: *"waiting 64 is what makes it tenable or it would be a constant hammering."*
 *
 * ⛳ TWO USES, NOT ONE, AND THE SECOND IS WHAT MAKES THIS DIAL EXPENSIVE. The quote is about the first: an
 * idle block polling its OWN base for work, which is `sys__computing_base__begin` backing off after a swap
 * it lost, and the engine's parked compute loop doing the same when it finds nothing. The second is the
 * inter-try delay of a BOUNDED spin on ANOTHER block's base while collecting a result — the compute
 * opcode's collect, and the dispatcher's in `engine/abi/running.cuh`.
 * ⛔ SO TURNING THIS MOVES A TIMEOUT. `SYS__COMPUTING_BASE__RESULT_TRIES` is priced THROUGH this number:
 * `contracts/objects/computing_base.cuh` prices one try — two compare-and-swaps and this wait — at
 * ~1.28 us on the host (`MEASURED`, as stated there), so its try count bounds the wait at about fifteen
 * seconds. Double this and that bound roughly doubles. ⇒ ★ IT IS THE ONE
 * ENTRY HERE THAT DOES NOT KEEP THE OPENING PROMISE — it cannot be different without something else being
 * different too — so whoever turns it re-derives the try count beside it.
 *
 * ⛔ AND IT DOES NOT VARY BY BLOCK, WHICH IS THE DIFFERENCE FROM THE OBJECT LOCK'S BACK-OFF. That one adds
 * the block id because its contenders share ONE WORD, and two that retry in step retry in step forever. An
 * idle block shares nothing — block N polls base N — so there is no rival to fall out of phase with
 * (⛳ and that stays true with runners: runner N polls base N, which is what `block_id` answering a
 * `__thread` slot is FOR), and
 * adding the id would only mean high-numbered blocks notice their work later than low-numbered ones for no
 * reason at all. A collector does read another block's word, and skew buys nothing there either: it is
 * waiting for ONE producer to finish, not racing a second collector. ⚖ *"this is not a deadlock risk but a
 * hammering risk ... we want it stable in this case."* Stable is the property being bought: every block sees
 * work after the same delay. */
#ifndef SYS__COMPUTING_BASE__POLL_CYCLES
#define SYS__COMPUTING_BASE__POLL_CYCLES 64u
#endif

/* How long a caller waits for an object's lock before giving up.
 *
 * ⛳ THERE IS A BOUND AT ALL BECAUSE AN UNBOUNDED SPIN IS THE ONE SHAPE THIS ENGINE CANNOT AFFORD:
 * nothing in the runtime notices a wait that never ends, so a holder that never lets go is a process
 * that hangs with no suspect. A bound turns that into a fault with a name.
 *
 * `REASONED`, not `MEASURED` — it is a guess at "longer than any legitimate holder", and no lock here has
 * met a second contender yet. ⛔ THAT IS A PROPERTY OF WHAT IS LOCKED AND NOT OF THE MACHINE, WHICH IS THE
 * PART A READER WILL GET BACKWARDS. The device runs many blocks at once — `reside`, `dispatch` and `grid`
 * all go out over the resolved block count, and the selftest boots four, then places work on three of them
 * and on block zero while the launch is up — but every caller of `sys__heap_object__lock` is a stack verb,
 * and a stack belongs to one block. ⇒ ★ THE GUESS IS UNTESTED BECAUSE NOTHING LOCKED IS SHARED, SO THE DAY
 * AN OBJECT IS SHARED BETWEEN BLOCKS IS THE DAY IT FIRST CARRIES WEIGHT. What would falsify it: a legitimate
 * holder that takes longer, which would show up as `OBJL`-tagged faults once something is contended.
 * ⛔⛔ AND A CPU EVALUATOR IS WHERE THAT DAY COMES FROM. ⚖ *"sys and the evaluator go host"* puts several
 * RUNNER THREADS in one address space over one heap — `cpu/silicon/atomic.cuh` has real atomics for
 * exactly that reason. The premise *"nothing locked is shared"* rests on every `sys__heap_object__lock`
 * caller being a stack verb and a stack belonging to one block — and a runner thread is the host's
 * block, so the shape may hold there too. **It has not been checked.**
 * ⇒ ★★ THIS ENTRY NAMES ITS OWN FALSIFIER, AND MULTI-RUNNER HOST EVALUATION IS WHERE IT WILL MEET IT —
 * which makes it the most valuable kind of comment and the easiest to walk past, because nothing about
 * it looks wrong. ⚠ `LOCK_TRIES` 64
 * is a guess that has never met a contender; the first genuinely shared object between runners is what
 * prices it. **UNMEASURED, and named here rather than discovered as an `OBJL` storm.** */
#ifndef SYS__HEAP_OBJECT__LOCK_TRIES
#define SYS__HEAP_OBJECT__LOCK_TRIES 64u
#endif

/* How long it waits between those tries.
 *
 * ⚖ ARCHITECT: *"make it wait 10 clock cycles plus its block id, so waits happen at different speed and
 * we resolve a potential deadlock."* The block id is added at the call site in `sys__heap_object__lock` and
 * is NOT a dial — it is the intent, and two contenders backing off by the same amount retry together
 * forever. This is the base it is added to.
 *
 * ⛳ ON THE HOST THE SKEW SURVIVES THE PRIMITIVE: `cpu/silicon/atomic.cuh`'s `sys__silicon__wait_cycles`
 * issues exactly `n` pauses, so every block id backs off by a different amount. `REASONED` from reading
 * it; nothing has timed it. ⛔ A backend whose wait rounds to a coarse unit would flatten the skew, and
 * it would say so beside its own definition.
 * ⇒ ★ SO WHAT BREAKS A LOCKSTEP RETRY WHERE THE SKEW CANNOT IS THE TRY COUNT ABOVE: the lock gives up
 * and faults with a name. De-phasing is the cheap win on a target that can express it; the bound is the
 * property that holds when it cannot. */
#ifndef SYS__HEAP_OBJECT__LOCK_BACKOFF_CYCLES
#define SYS__HEAP_OBJECT__LOCK_BACKOFF_CYCLES 10u
#endif

/* How long a carousel taker waits for the adder that reserved the slot before it to finish writing, and
 * how long each of those waits is.
 *
 * ⛳ THE WAIT DOES NOT VARY BY BLOCK, WHICH IS THE DIFFERENCE FROM THE LOCK'S BACK-OFF ABOVE. That one adds
 * the block id because its contenders share ONE WORD and two that retry in step retry in step forever.
 * Nobody is contending here: the taker owns its slot outright and is waiting for ONE adder to finish a
 * store it has already committed to. */
#ifndef SYS__CAROUSEL__WAIT_TRIES
#define SYS__CAROUSEL__WAIT_TRIES 64u
#endif
#ifndef SYS__CAROUSEL__WAIT_CYCLES
#define SYS__CAROUSEL__WAIT_CYCLES 10u
#endif

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_DEFAULTS_CUH */
