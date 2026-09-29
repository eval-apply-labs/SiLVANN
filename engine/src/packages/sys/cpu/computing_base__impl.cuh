#ifndef SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/kind.cuh"              /* a result carries its kind, so the slot is read back as one */
#include "heap__header.cuh"      /* finishing lets the running copy go, which is the heap's verb */
#include "heap_object__header.cuh"
#include "computing_base__header.cuh"
#include "silicon/silicon__header.cuh"
#include "list__header.cuh"      /* a program arrives as a picture, and `begin` thaws one */
#include "bindings__header.cuh"  /* an environment is built over a snapshot, and the reach is armed */
#include "bindings__impl.cuh"    /* the bindings reach, and an environment over a snapshot */
#include "node_array__header.cuh" /* a snapshot of an environment is one, and is told apart by kind */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * THE LISP SURFACE THIS FILE OWES — and this one retires when it is BUILT, not when the port ends.
 *
 * The three verbs below are published: `language_contract.cuh` registers `sys__compute`, `sys__result`
 * and `sys__completed`, so a program calls them by name and the C below is what they land in. ⛳ RE-DERIVE
 * THE ROWS RATHER THAN TRUSTING THIS SENTENCE:
 *   grep -n 'sys__compute\|sys__result\|sys__completed' \
 *        packages/sys/contracts/macros/language_contract__verbs.cuh
 *   (`language_contract.cuh` only includes that file, so a grep of it finds nothing.)
 * ⇒ THE RETIREMENT CONDITION AT THE FOOT OF THIS BLOCK IS MET — every row it lists as owed is BUILT.
 * ⚖ ARCHITECT'S NOTATION: `(compute (cu) (bindings) (actions))`.
 *
 * ── WHAT IS OWED ────────────────────────────────────────────────────────────────────────────────────
 *   compute(cu, bindings, actions)   ✅ BUILT. `claim` and then `init`, composed — never either alone,
 *                                    because a claim without a fill leaves a base held by nobody and a
 *                                    fill without a claim is the race `init` refuses.
 *   completed?(cu)                   ✅ BUILT as `sys__completed`. The blocker this row named — *"there is
 *                                    no boolean object in `src`"* — stopped being true when the truth
 *                                    kinds landed, and the verb answers `sys__opcodes__truth` like every
 *                                    other predicate. ⛳ NAMED WITHOUT THE `?`, matching `sys__eq`.
 *   result(cu)                       ✅ BUILT. Once-only is part of the surface rather than an
 *                                    implementation detail: the second caller is handed nothing, and a
 *                                    program can see that. ⛳ IT STILL WAITS, BOUNDED, which is now a
 *                                    convenience rather than the only way to find out — `sys__completed`
 *                                    is what a program polls with.
 *
 * ── WHAT DOES NOT NEED ONE, AND WHY, SO IT IS NOT ADDED LATER OUT OF SYMMETRY ───────────────────────
 *   status      subsumed. `completed` says whether it is over, and an errored computation hands back an
 *               ERROR OBJECT — so a program asks the RESULT what happened, not the base. That is the
 *               whole payoff of an error being a value.
 *   begin       the block's own idle loop, below the language: `while (b == 0) b = begin(me);`. A program
 *               that could call it could start on somebody else's work.
 *   claim       never alone — see `compute`.
 *   finish      the runtime says how a computation went; a program saying so could lie about its own
 *               outcome.
 *   advance     private already, and the reason is written where it is defined.
 *
 * ⭐ AND ONE SHAPE DECISION THAT BELONGS HERE RATHER THAN IN THE OPCODES: A PROGRAM NAMES A CU, NEVER A
 * BASE. `(compute (cu) ...)` takes an index, and the base stays machinery the language cannot hold. That
 * is not squeamishness — a computing base is NEVER RELEASED, so handing one out as a value would put an
 * object into the language that every lifetime rule in this package has an exception for. An index has no
 * lifetime at all.

 * ── WHAT CHANGED, TAKEN OUT OF THE ORDINARY COMMENTS ────────────────────────────────────────────────
 *   · `begin` could not fail until the thaw moved into it; the ERROR arms at the thaw and the
 *     environment are what that move added.
 *   · `init` once copied both operands and rolled the first back when the second failed; it retains now.
 *   · `init` once blanked the providers too; that moved into `zzpackage_start`, the one caller that must.
 *   · the generic state store was once package-reachable; `zzpackage_start` is what let it go private.
 *   · the status was once the catch; an error VALUE now arrives at OK and the status only says "unsound".
 *
 * RETIREMENT: delete when the lisp surface below is built. Every line above names something this file
 * OWES rather than something it does, so the block empties as the debt is paid.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛔ THE ONE THING `init` PRODUCES THAT IT IS NOT HANDED IS A WORD, AND NOT AN OBJECT. A base that cannot
 * be filled parks the FAULT CODE `SYS__COMPUTING_BASE__FAULT_HALF` in the PROGRAM row and raises, and
 * `sys__computing_base__fault` reads it back without taking anything — which is why nothing is declared
 * here and why `error.cuh` is not among the includes above. ⚖ The argument is the one `begin` makes at
 * its own failing carve: a description that must ask the heap for room would fail exactly when the heap is
 * what refused, leaving the slot at zero, which is the one answer that carries no detail at all.
 *
 * ⛳ AN ERROR IS STILL A VALUE, AND THAT DOCTRINE IS UNTOUCHED — it is about a program that EVALUATED to
 * one, which comes back through `read_result` like any other result. A refused `init` never started a
 * program, so there is nothing for it to be the value of. */

/* ⛔ AND THIS ONE IS TRANSITIONAL, WHICH IS WHY IT IS HERE RATHER THAN BESIDE ITS CALLER. It is the only
 * name in this file that belongs to nobody who will keep it.
 *
 * It reads the per-block array of bases, and that array is the SYSTEM REGISTER's. The boot hands its room
 * to `sys__system_register__publish`, which places it and writes the row
 * `SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY`; `sys__heap__publish` is handed no such table at all.
 * ⚖ RULED, AND BUILT — the body is a register read plus an index, which is what the ruling said this
 * becomes. What is still owed is the NAME. The accessor sits with the heap because the allocator asks the
 * same question on its own way to every allocation, so a `sys__heap__` prefix names where the code lives
 * rather than who owns what it reads.
 *
 * ⛳ IT IS WRITTEN DOWN AS A DEBT RATHER THAN LEFT TO READ AS A DESIGN, because a `sys__heap__` name
 * sitting in this file's promises otherwise says the heap is where bases live — which is the thing the
 * ruling changed. */
static __device__ inline sys__heap_node* sys__heap__zzpackage_computing_base(uint32_t block_id);

/* Whether this is one, asked of the node the allocation begins with rather than of the node itself. That
 * is where every kind of object says what it is, so the counting and this ask the same question of the
 * same place, and there is one answer rather than two that could differ.
 *
 * Asked before anything is read out, because everything below is arithmetic on whatever the node happens
 * to hold and there is no other way to notice being handed the wrong one. */
static __device__ inline bool sys__computing_base__is(const sys__heap_node* computing_base) {
    /* ⭐ ONE LINE, AND IT IS THIS FILE'S SURFACE RATHER THAN ITS MECHANISM. The step back to the allocation
     * head, and the null test guarding that arithmetic, belong to `sys__heap_object__type`, which the call
     * below reaches through `sys__heap_object__is_type`, because *"an object pointer is one node past its
     * head"* is a fact about the LAYOUT and not about computing bases. What is left is the kind, which IS
     * this file's to know. */
    return sys__heap_object__is_type(computing_base, SYS__KIND__COMPUTING_BASE);
}

/* Read and write one of the two providers. Package-only, and the marker is a description rather than a
 * wish: the HEAP is the only caller there is. `make` reads the slot the kind's row names to find the chunk
 * to carve from and writes it when the block moves to the next one, and `start_block` writes the first.
 * Code that allocates says `make(kind)` and never learns which provider answered — which is the whole
 * point of deriving it from the row, and the reason nothing outside this package has business here.
 *
 * ⛔ THE OTHER FOUR SLOTS ARE NOT REACHABLE THIS WAY, AND WHAT KEEPS EACH OF THEM STRAIGHT IS NAMED. A
 * provider holds a RAW chunk offset with nothing counted against it, so overwriting one loses nothing.
 * The bindings and the program hold counted REFERENCES — overwriting one drops a hold that is never given
 * up, so `init` fills them and `read_result` takes them back and nobody else does either. The status is a
 * STATE: storing into it walks around the swap that makes a claim exclusive, so only the transitions move
 * it. The result kind is the one with TWO writers — `finish` stamps it and the drain clears it — and those
 * are the two ends of one computation rather than two parties, which is why a pair is not a contention.
 * Refusing all four here is what makes those sentences true rather than expected.
 *
 * ⛳ NEITHER TAKES A LOCK. ⚖ ARCHITECT: *"everything but the status will be dealt in a single threaded
 * way."* The providers belong to the one block that owns the base; the status is the one word two blocks
 * can reach, and it is the one word that swaps.
 *
 * ⭐ THE VERB SAYS `heap_provider` AND NOT JUST `provider`, AND THE REASON IS THE ONE THING THIS NODE
 * MAKES HARD. ⚖ ARCHITECT: *"should they not be sys__computing_base__zzpackage_get_heap_provider?"* —
 * yes, because the base holds TWO KINDS OF OFFSET IN ONE TYPE and only the slot tells them apart. The
 * providers hold a CHUNK offset, which is a multiple of the chunk size; bindings and program hold a
 * REFERENCE, which is a node inside one. Both are `uint64_t`, the two sets are disjoint by construction —
 * `addressable` is where that is tested, and it tests POSITION WITHIN A CHUNK rather than a remainder,
 * because kinds choose their own width and a remainder would assume they did not. Nothing in the
 * signature said which of the two came back.
 * ⇒ ★ WHEN ONE TYPE CARRIES TWO MEANINGS, THE NAME IS THE ONLY PLACE THE DIFFERENCE CAN LIVE. It is
 * cheaper here than a tag beside every value, and it is the reason the file argues the two sets are
 * disjoint rather than tagging them.
 *
 * ⚠ AND THE ADJACENT NAME IS NOT THE SAME THING, WHICH IS EASY TO READ WRONG NOW THAT THIS ONE IS LONGER:
 * `sys__heap__provider(kind)` answers WHICH SLOT a kind is carved from — an index, `OBJECTS` or `ARRAYS`.
 * These answer WHAT IS IN a slot. One is a question about a kind, the other about a base. */
static __device__ inline uint64_t sys__computing_base__zzpackage_get_heap_provider(const sys__heap_node* computing_base,
                                                                 uint32_t which) {
    if (which > SYS__COMPUTING_BASE__ARRAYS) return 0ull;
    return sys__computing_base__is(computing_base) ? computing_base->args[which] : 0ull;
}

static __device__ inline void sys__computing_base__zzpackage_set_heap_provider(sys__heap_node* computing_base, uint32_t which,
                                                             uint64_t offset) {
    if (which > SYS__COMPUTING_BASE__ARRAYS) return;
    if (sys__computing_base__is(computing_base)) computing_base->args[which] = offset;
}

/* Set the computing state. Its callers already have the exclusion a swap would buy,
 * and they have it for DIFFERENT reasons:
 *
 *     `zzpackage_start` a base being carved has never been published. Nothing knows where it is, so there
 *                     is nobody to exclude and nothing to swap against.
 *     `init`          its caller CLAIMED the base first — the state already names that caller, and this
 *                     is checked, not assumed. Swapping here would be swapping against itself.
 *     `begin`         it swapped the base out of SCHEDULED on the way in, so this block is already the
 *                     only one that may touch the word. Its two ERROR arms park a start that could not
 *                     carve, and there is nobody left to race for it.
 *
 * That is the whole of the licence, and it is why the marker is `zzprivate`: a caller that has not already
 * excluded everybody has no business with this and must transition instead. */
static __device__ inline void sys__computing_base__zzprivate_set_status(sys__heap_node* computing_base,
                                                                        uint32_t state) {
    if (sys__computing_base__is(computing_base))
        computing_base->args[SYS__COMPUTING_BASE__STATUS] = (uint64_t)state;
}

/* ── HOW THE COMPUTATION IS GOING, AND WHO GETS TO SAY SO ────────────────────────────────────────────
 * 
 * Return the current status of the computation
 * It needs to not have a lock otherwise a computing state would never return
 *
 * ⛔ IT SAYS NOTHING ABOUT WHICH STATES MAY FOLLOW WHICH, AND THAT IS DELIBERATE UNTIL THERE IS A
 * SCHEDULER. A caller names both ends, so the legal moves live at the call sites that make them rather
 * than in a table here that nothing yet reads. It becomes a real question when a queue arrives.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint32_t sys__computing_base__status(const sys__heap_node* computing_base) {
    return sys__computing_base__is(computing_base)
         ? (uint32_t)computing_base->args[SYS__COMPUTING_BASE__STATUS]
         : SYS__COMPUTING_BASE__FREE;
}

/* Move it from one state to another, and answer whether you were the one who moved it. Every transition
 * goes through here, so there is one place a state can change and it is a compare-and-swap.
 *
 * ⭐⭐ WHY IT TAKES `from` AT ALL, WHICH IS THE QUESTION TO ASK OF IT: because a plain set could not say
 * WHO IS ASKING, and the same destination is reached by two different callers with two different rights.
 *
 *     an outsider  swaps from FREE       -- "this base is idle and I am taking it to fill"
 *     the owner    swaps from SCHEDULED  -- "this base was filled for me and I am starting"
 *
 * Both end at COMPUTING + their own id. A `set` reaches that from anywhere, so an outsider could seize a
 * base mid-computation and the owner could start on one nobody had filled — and neither would be
 * detectable afterwards, because the word would look exactly as it should.
 *
 * ⇒ ⭐ SO THE `from` IS NOT A PRECONDITION BEING CHECKED, IT IS THE PERMISSION BEING SPENT. A caller says
 * which right it is exercising by naming the state that right starts from, and the swap either finds that
 * state and consumes it or finds another and refuses. That the check and the move are one instruction is
 * what makes the right unforgeable rather than merely tested.
 *
 * ⛔⛔ AND THAT IS ONLY TRUE WHILE THIS IS PRIVATE, WHICH IS WHY IT IS. A caller that can name both ends
 * can build an edge nobody designed — seize a base that is mid-computation, or declare a finished one
 * scheduled again — and the word afterwards looks exactly as it should, so nothing downstream can tell.
 * Left public, the argument above describes a convention; file-private, the four verbs below are the only
 * `from` anybody can name, and it describes the code.
 *
 * ⛳ WHICH IS ALSO WHY THEY ARE FOUR VERBS AND NOT ONE `set_state`. They differ ONLY in the `from` they
 * name, so choosing the verb IS naming who you are — and a caller cannot exercise a right it does not
 * have, because there is no argument through which to claim one. It can only fail to move the base. */
static __device__ inline bool sys__computing_base__zzprivate_advance(sys__heap_node* computing_base,
                                                           uint32_t from, uint32_t to) {
    if (!sys__computing_base__is(computing_base)) return false;
    unsigned int* word = (unsigned int*)&computing_base->args[SYS__COMPUTING_BASE__STATUS];
    return sys__silicon__cas_u32(word, (unsigned int)from, (unsigned int)to) == (unsigned int)from;
}

/* Ask to schedule work on a computing base, and be told yes or no. Winning writes this block's identity 
 * into the state, so it is held and known to be held by you in the same instruction.
 *
 * ⛔⛔ IT IS THE FILL EDGE, AND IT IS THE ONLY CONTENDED ONE. ⚖ ARCHITECT: *"the atomiccas is on the
 * computing_base, if the claim returns true for someone it locks it so no one else can read it, the
 * second one would see an atomiccas value different from zero and return 0 instead of the computing
 * base."* Any block may schedule work onto any other block's base, so the taking has to be a swap from
 * FREE — and FREE is zero, so the second caller finds a value that is not zero and is handed nothing.
 * Filling is then exclusive by construction rather than by whoever happens to call it.
 *
 * ⛔⛔ A NO IS FINAL AND THAT IS THE WHOLE CONTRACT. ⚖ ARCHITECT: *"my worry is that calling it when locked
 * means that callers will keep hammering, which is bad. i need a way to refuse the connection
 * gracefully."* THERE IS NO RETRY HERE AND THERE MUST NEVER BE ONE. A failed swap means somebody else is
 * running a whole computation, which is not a wait anybody should sit through — the caller goes and finds
 * other work. That is the difference from `sys__heap_object__lock`, which spins sixty-four times before it
 * gives up, because a lock has nothing to tell a loser except "not yet".
 *
 * ⛳ AND IT IS WHY THIS IS A `claim` AND NOT A `lock`. The word a caller reads at the call site should say
 * that being refused is an ordinary answer rather than a failure to handle.
 *
 * ⭐ IT ANSWERS WITH THE BASE ITSELF, NOT WITH A YES. ⚖ ARCHITECT: *"the return value of claim should be
 * the computing_base or null/zero on the handshake."* Permission and the thing permitted are then ONE
 * value, so there is no arrangement in which a caller has been told yes and does not have what it may use
 * — the same reason the state names its holder rather than saying "held". It reads as the handshake it is:
 *
 *     sys__heap_node* mine = sys__computing_base__claim(candidate, my_block);
 *     if (mine != 0) { ... }
 *
 * ⛳ And when a claim is one day made against a HEAP ALLOCATION POOL rather than a base someone already has, the answer is
 * already the right shape: which one you got, or none. */
static __device__ inline sys__heap_node* sys__computing_base__claim(sys__heap_node* computing_base,
                                                                uint32_t block_id) {
    return sys__computing_base__zzprivate_advance(computing_base, SYS__COMPUTING_BASE__FREE,
                                        SYS__COMPUTING_BASE__COMPUTING + block_id)
         ? computing_base : (sys__heap_node*)0;
}

/* The owner begins computing on what somebody scheduled for it. SCHEDULED is the trigger, and taking it
 * is how a block says it is now the one running — ⚖ ARCHITECT: *"scheduled is the trigger to load the
 * execution list and bindings and declare itself as computing."*
 *
 * ⛳ IT IS A SWAP THOUGH NOTHING CONTENDS FOR IT, AND THE REASON IS WORTH KEEPING: only block N ever runs
 * base N, so this edge has no rival the way the fill does. It swaps anyway because a store cannot notice
 * that the base was not actually scheduled — a runner woken by a stale read would begin on a base nobody
 * filled, and that reads as working right up until it evaluates whatever the last tenant left.
 *
 * ⛔⛔ AND A NO HERE IS NOT A NO LIKE `claim`'s, WHICH IS WHY THIS ONE WAITS AND THAT ONE MUST NOT. A block
 * refused a claim has somewhere else to be — the base it wanted is busy for a whole computation, so it
 * goes and finds other work. A block refused a BEGIN has nowhere else: it IS the idle loop, "not yet" is
 * the answer it expects most of the time, and it will ask again whatever this returns. So the back-off
 * belongs in here rather than in every caller that would have to remember it — without one, an idle block
 * polls a word at full rate, and the traffic is the entire cost of having nothing to do.
 *
 * ⛔ IT DOES NOT LOOP AND IT DOES NOT RAISE. `sys__heap_object__lock` gives up after a count of tries and
 * faults, because a lock held that long is a bug. A base that is not scheduled is not a bug, it is the
 * ordinary state of an idle block — so this waits once and answers, and how long a runner keeps asking stays
 * the caller's business rather than a number buried here.
 *
 * ⛳ THE DELAY IS THE SAME FOR EVERY BLOCK, AND IT IS NOT THE LOCK'S BACK-OFF. That one adds the block id
 * because its contenders share one word; these share nothing, since block N polls base N — so there is no
 * rival to fall out of phase with, and skewing would only make high-numbered blocks notice their work
 * later for no reason. The number itself lives in `contracts/defaults.cuh`, because it is a dial and not a
 * conclusion. */
static __device__ inline sys__heap_node* sys__computing_base__begin(sys__heap_node* computing_base,
                                                                uint32_t block_id) {
    if (!sys__computing_base__zzprivate_advance(computing_base, SYS__COMPUTING_BASE__SCHEDULED,
                                                SYS__COMPUTING_BASE__COMPUTING + block_id)) {
        sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
        return (sys__heap_node*)0;
    }

    /* ⭐⭐ AND HERE IS WHERE THE PROGRAM BECOMES THIS BLOCK'S OWN. What was scheduled is a PICTURE — plain
     * arrays that nobody may write, which is what let one of them be handed to every block at once. Running
     * it means rewriting it, so each block needs a list of its own, and this is the moment it gets one.
     *
     * ⭐ IT IS THAWED HERE AND NOT IN `init` FOR ONE REASON, AND IT IS THE WHOLE POINT OF THE ARRANGEMENT:
     * a thaw carves from the arena of WHOEVER CALLS IT. `init` is run by whoever schedules, so thawing
     * there would put every block's program in the scheduler's chunks — and it would still run, correctly,
     * with the locality thrown away and nothing anywhere to show it.
     * ⛳ WITH RUNNER THREADS IT IS CONTENTION AS WELL: several threads claiming from the scheduler's
     * chunks at once. Same fix either way. ⚠ UNMEASURED: every program so far runs on block 0 with the
     * others parked, so nothing has contended here yet.
     *
     * ⛔ THE PICTURE'S HOLD GOES BACK EITHER WAY, and it goes back AFTER the thaw rather than before: the
     * thaw reads it. This base took that hold in `init` so the picture could not go between being scheduled
     * and being started; once there is a copy, the base has no further use for it. Whoever dispatched still
     * holds their own, so letting this one go does not take the picture out from under the other blocks
     * that have not started yet. */
    const uint64_t picture = computing_base->args[SYS__COMPUTING_BASE__PROGRAM];
    /* ⭐ THE EVALUATOR'S DOOR — this copy is read by `eng__eval` and by nothing else. ▶ the two
     * doors in `list__impl.cuh`: the host's stays eager because nothing out there knows the tag. */
    const uint64_t run     = sys__list__thaw_running(picture);
    (void)sys__heap_object__release(picture);

    /* ⛔⛔ SO BEGINNING CAN FAIL — a state swap and a back-off cannot come up short, and an allocation can.
     * The answer is the one `init` gives when it refuses: the FAULT CODE goes in the PROGRAM row and the
     * base goes to ERROR, where `sys__computing_base__fault` reads it back. One more arrow, and no new
     * state.
     *
     * ⛳ AND IT ANSWERS ZERO TO THE OWNER — the same answer "nothing is scheduled yet" gives. That is
     * deliberate rather than convenient: an owner has nothing to run in either case, so it goes back to
     * asking, and the base is not stranded because ERROR is collectable. The requestor drains it and the
     * base comes back to FREE like any other finished one.
     *
     * ⛳ THE STATUS IS STORED RATHER THAN SWAPPED, because the swap above already made this block the only
     * one that may touch it — which is the same reason `init` stores it. */
    if (run == 0ull) {
        /* ⛔ THE CODE GOES IN THE SLOT AND NOTHING IS BUILT, because what just failed was a carve. An
         * error OBJECT would ask the heap for room at the moment it has refused, so the description of
         * the failure would fail in the same way — leaving the slot at zero, which is the one case that
         * carries no detail at all. A fault is a word and the slot is a word. */
        computing_base->args[SYS__COMPUTING_BASE__PROGRAM] = SYS__COMPUTING_BASE__FAULT_NO_THAW;
        sys__fault__raise(0ull, SYS__COMPUTING_BASE__FAULT_NO_THAW);
        sys__computing_base__zzprivate_set_status(computing_base, SYS__COMPUTING_BASE__ERROR);
        return (sys__heap_node*)0;
    }
    computing_base->args[SYS__COMPUTING_BASE__PROGRAM] = run;

    /* ⭐⭐ AND THE ENVIRONMENT IS MADE HERE TOO WHEN WHAT WAS SCHEDULED IS A SNAPSHOT, FOR THE REASON THE
     * PROGRAM IS. A read-only array is what an environment is BUILT OVER — a read falls through to it and
     * a write never does — so many blocks can share one with nothing coordinating them, and each gets its
     * own scopes on top. Building those scopes is an allocation, and an allocation belongs in the arena of
     * the block that will read it, which is this one and not the scheduler's.
     * ⭐ IT IS NEARLY FREE BY THE DESIGN'S OWN CONSTRUCTION: a scope is an empty stack and an empty stack
     * claims no chunk until something is pushed onto it, so an environment over n names is n granules out
     * of a chunk this block is already carving from.
     *
     * ⛔⛔ AND BEING HANDED AN ENVIRONMENT MEANS SOMETHING DIFFERENT AND IS LEFT ALONE. A caller that
     * schedules the environment ITSELF is saying the computation binds into theirs — which is what makes
     * `defun` here and a call to it later the same environment, and there is exactly one block involved.
     * A caller that schedules a SNAPSHOT is saying the opposite. The two are told apart by kind because
     * they are two different intentions, not two spellings of one. */
    const uint64_t given = computing_base->args[SYS__COMPUTING_BASE__BINDINGS];
    if (given != 0ull && sys__node_array__is(given)) {
        /* A snapshot is as wide as the environment it was taken of, plus the hatch at its position 1. */
        const uint64_t mine = sys__bindings__create(sys__node_array__length(given) - SYS__BINDINGS__FIRST_NAME,
                                                    given);
        if (mine == 0ull) {
            (void)sys__heap_object__release(run);
            computing_base->args[SYS__COMPUTING_BASE__PROGRAM] = SYS__COMPUTING_BASE__FAULT_NO_ENV;
            sys__fault__raise(0ull, SYS__COMPUTING_BASE__FAULT_NO_ENV);
            sys__computing_base__zzprivate_set_status(computing_base, SYS__COMPUTING_BASE__ERROR);
            return (sys__heap_node*)0;
        }
        /* The snapshot is held by the environment now; what `init` took for the base is given back here
         * for the same reason the picture's is above — this base has no further use for it once there is
         * something built over it, and the other blocks that have not started yet hold their own. */
        (void)sys__heap_object__release(given);
        computing_base->args[SYS__COMPUTING_BASE__BINDINGS] = mine;
    }

    /* And the reach, on the environment this block will actually read names out of — which is only known
     * here, because it is either what was scheduled or the one just built over a snapshot. It is an
     * optimisation and it heals itself when it does not match, so being armed late costs a lookup and
     * never an answer; being armed on somebody else's environment would cost the answer. */
    sys__bindings__zzpackage_reach_arm(computing_base->args[SYS__COMPUTING_BASE__BINDINGS]);
    return computing_base;
}

/* Put down what you took, at OK or at ERROR, and leave behind what it came to. It names the block so that
 * only the holder can end it: a caller passing an id that is not the one in the word does not match, and
 * nothing moves.
 *
 * ⛔⛔ THE STATE IS READ AND THEN ACTED ON, WHICH IS THE PATTERN THIS PACKAGE OTHERWISE REFUSES, AND HERE
 * IT IS SOUND FOR A REASON WORTH WRITING DOWN. Everywhere else a check-then-act loses because a second
 * party can move the word in between. Nothing can move it here, and the property that buys that is not
 * scarcity of writers — FIVE ROUTES REACH COMPUTING+N — it is that every one of them writes the id of
 * whoever acts next. `claim` and `begin` leave it naming the block that took it. `read_result`'s two
 * swaps leave it naming the collector, which is already inside the call that empties the base.
 * `zzpackage_start` stores it plainly on a node nothing has been told the address of. And the host, in
 * `eng_abi_result_release`, names the block it is handing the base back to. Moving it away again is this
 * verb, `init`'s stores (SCHEDULED, FREE, or ERROR on a half fill) and `begin`'s two ERROR arms, and all
 * of them run on a base the word already names the caller in. So a base
 * seen at COMPUTING+me can only be moved by me, and reading it first is not a window — the word is its
 * own permission.
 * ⇒ ★ THE GUARD THAT MAKES A CHECK SAFE IS EXCLUSIVITY, NOT ATOMICITY, and this word carries its owner.
 *
 * ⛔ AND THE RUNNING COPY IS LET GO OF HERE, WHICH IS WHAT THE BLOCK THAT RAN IT OWES. `begin` carved a
 * private list out of this block's own arena and the base has held it ever since; the value is a cell
 * pointing into it, so the value's referent is taken FIRST and the list let go after — the other order
 * frees the answer along with the room it sat in. */
static __device__ inline bool sys__computing_base__finish(sys__heap_node* computing_base, uint32_t block_id,
                                                          uint32_t outcome, sys__heap_node value) {
    if (!sys__computing_base__is(computing_base)) return false;
    if (sys__computing_base__status(computing_base) != SYS__COMPUTING_BASE__COMPUTING + block_id)
        return false;

    /* ⛳ ONLY AT OK, AND THE ERROR PATH IS WHY. A base that failed carries a fault WORD in this same slot
     * rather than an offset — releasing it would take a hold off whatever object happens to live there,
     * and the word is what `fault` exists to read back. So a failed computation leaves the slot exactly
     * as whoever failed it wrote it. */
    /* ⭐⭐ AND WHAT THE COMPUTATION USED IS LET GO OF AFTER THE ANSWER IS OUT, NOT BEFORE. ⚖ *"i am ok with the
     * result being allowed immediately with ok and the drain happening after the ok"*. The program's running
     * copy and the environment it ran in are taken out of the base here and released once the status says OK,
     * so whoever is waiting for the answer does not wait for the teardown too — and the environment is released
     * by the block that ran in it, not by the one collecting, which is on its own way to the next hand-off.
     * ⛳ THEY ARE THIS CALL'S ONCE TAKEN: the base names neither, so a collector drains nothing of them, and a
     *   base scheduled again the moment the OK lands has nothing of this computation in it. */
    uint64_t ran = 0ull, used = 0ull;
    if (outcome == SYS__COMPUTING_BASE__OK) {
        if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__retain(value.args[0]);
        ran  = computing_base->args[SYS__COMPUTING_BASE__PROGRAM];
        used = computing_base->args[SYS__COMPUTING_BASE__BINDINGS];
        computing_base->args[SYS__COMPUTING_BASE__BINDINGS]    = 0ull;
        computing_base->args[SYS__COMPUTING_BASE__PROGRAM]     = value.args[0];
        computing_base->args[SYS__COMPUTING_BASE__RESULT_KIND] = (uint64_t)value.dtype;
    }

    /* ⛔ AND THE ANSWER IS PUBLISHED BEFORE THE STATUS SAYS THERE IS ONE, for the reason the fill was on
     * the way in: whoever scheduled this is watching the status word, and a status that lands first is an
     * invitation to read an answer that has not. */
    sys__silicon__publish();
    const bool advanced = sys__computing_base__zzprivate_advance(computing_base,
                                                                 SYS__COMPUTING_BASE__COMPUTING + block_id, outcome);
    if (ran != 0ull) (void)sys__heap_object__release(ran);
    if (used != 0ull) {
        sys__bindings__zzpackage_reach_forget();   /* this block's cache of the environment going away */
        (void)sys__heap_object__release(used);
    }
    return advanced;
}

/* Whether it is over, either way. ⚖ ARCHITECT: *"a true/false method for completed that checks without a
 * lock if the status is ok or error."* Which of the two it was is `status`; this answers only whether
 * there is anything still to wait for, and reads one aligned word to do it.
 *
 * ⛳ A C `bool` IS THE RIGHT ANSWER TO A C CALLER, and it is not the raw-int predicate the project
 * forbids: that rule is about a value handed to `?` or `while` as truth, and what reads this one is a C
 * `if`. THE LISP SURFACE OF THIS ANSWERS `#t`/`#f`, and the opcode below is what makes the object:
 * `sys__opcodes__zzabi_completed` hands this predicate to `sys__opcodes__truth`, which
 * answers a `VALUE_TRUE` or a `VALUE_FALSE` — two kinds, no payload, so an answer's identity is its tag
 * and an integer handed to a test is a refusal rather than a truth.
 * ⇒ ★ THE BOUNDARY OWNS THE OBJECT, NOT THE PREDICATE. One C answer, wrapped where it crosses into the
 * language, is why there is one definition of what true is rather than two that could differ. */
static __device__ inline bool sys__computing_base__completed(const sys__heap_node* computing_base) {
    const uint32_t st = sys__computing_base__status(computing_base);
    return st == SYS__COMPUTING_BASE__OK || st == SYS__COMPUTING_BASE__ERROR;
}

static __device__ inline uint64_t sys__computing_base__fault(const sys__heap_node* computing_base) {
    if (computing_base == 0) return 0ull;
    if (sys__computing_base__status(computing_base) != SYS__COMPUTING_BASE__ERROR) return 0ull;
    return computing_base->args[SYS__COMPUTING_BASE__PROGRAM];
}

/* ── MAKING ONE READY TO RUN, OR READY FOR NOTHING ───────────────────────────────────────────────────
 * A base is filled with the two things a computation is: the names it may use and the program it is. Both
 * arrive as somebody else's and both STAY somebody else's — what this takes is a hold, not a copy.
 *
 * ⭐ ZEROING IS THE FIRST THING IT DOES, AND THAT IS WHY THERE IS NO SEPARATE BLANKING VERB. A base given
 * neither a program nor bindings is a base that holds nothing — which is the state a block starts in — so
 * `init(base, me, 0, 0)` IS the blanking, and both the carving path and `read_result` reach it through
 * this one door instead of repeating its three stores.
 *
 * ⛔ NOTHING IS RELEASED HERE, WHICH IS WHAT MAKES IT SAFE ON A NODE THAT WAS JUST CARVED. The heap allocation pool does
 * not zero what it hands out, so every word holds whatever the chunk's last tenant left; reading those as
 * references would take a count on whatever offset the leftover bytes happen to spell. This OVERWRITES and
 * never releases. Giving up what a LIVE base holds is `read_result`, which releases what was lent, hands
 * over what was produced, and then comes here.
 *
 * ⭐⭐ THE TWO ARGUMENTS ARE THE SAME KIND OF THING TO THIS VERB, AND THAT IS WHAT MAKES IT SHORT. Each is
 * something IMMUTABLE that many bases may be given at once, so each is taken the same way: a count goes up
 * and nothing is carved. There is no copy here to succeed or fail, and no kind is asked what it is.
 * The BINDINGS are an environment's base — the tier underneath every scope, shared by everyone who can see
 * those names.
 * The PROGRAM is a PICTURE of one: plain arrays that nobody may write, produced ONCE by whoever dispatches.
 *
 * ⭐ AND THE COPY BELONGS TO `begin` RATHER THAN TO THIS VERB, WHICH IS A QUESTION OF OWNER AND NOT OF
 * ORDER. A copy made here would be carved from the SCHEDULER's arena; made in `begin` it is carved from
 * the arena of the block that will run it. Both work. Only one of them is local, and nothing downstream
 * can tell which you got.
 *
 * ⛳ IMMUTABILITY IS ALSO WHY NOTHING HERE TAKES A LOCK — ⚖ ARCHITECT, *"there are no locks to be had"* —
 * with retain and release still atomic, because the COUNT moves even when the contents cannot.
 *
 * ⛳ BOTH OR NEITHER, AND HALF IS REFUSED BEFORE ANYTHING IS TAKEN. A base naming a program and no bindings
 * is a state nothing downstream has a way to check for, so it is not reachable through here. ⛳ AND THAT
 * CHECK IS THE WHOLE OF THE GUARANTEE: a retain cannot come up short, so there is no case where one was
 * taken and the other refused, and nothing to roll back.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline bool sys__computing_base__init(sys__heap_node* computing_base, uint32_t block_id,
                                                        uint64_t bindings_viewonly,
                                                        uint64_t program_picture) {
    if (!sys__computing_base__is(computing_base)) return false;

    /* ⛔⛔ YOU MAY ONLY FILL A BASE YOU HAVE CLAIMED, AND THIS IS WHERE THAT STOPS BEING A CONVENTION. The
     * state must say COMPUTING and it must name YOU — not merely "somebody holds it", which would let a
     * second block fill what a first one claimed and lose its clones. `claim` is the only way to get here
     * legitimately, so the claim is mandatory rather than expected.
     *
     * ⛳ AND IT COVERS EVERY OTHER STATE A BASE CAN BE IN. A base that is SCHEDULED, or running, or
     * FINISHED-but-not-cleared, is not held by the caller, so it is refused for the same reason —
     * including the case that matters most: a finished base still HOLDS a program and bindings, and the
     * blanking below does not release, so filling one on top of them would leak both with every count
     * reading healthy. Getting a used base back to FREE is the drain behind `read_result`, and that is
     * the only route. */
    if (sys__computing_base__status(computing_base) != SYS__COMPUTING_BASE__COMPUTING + block_id) {
        sys__fault__raise(0ull, SYS__COMPUTING_BASE__FAULT_BUSY);
        return false;
    }

    /* ⛔⛔ THE TWO REFERENCES ONLY. THE PROVIDERS ARE NOT TOUCHED, AND THAT IS A LIFETIME AND NOT A TIDINESS
     * CHOICE. A provider names a heap chunk THIS BLOCK IS CARVING INTO, and a block does not stop carving
     * between computations — ⚖ ARCHITECT: *"you then need to remove it when you get a new chunk, not when
     * you exit from compute otherwise you lock the release of that heap chunk."* Zeroing them does not
     * give a chunk back; it only makes the block forget it owns one, so the next allocation claims a fresh
     * chunk and the old one is stranded with nobody to empty it. ⛳ `MEASURED`: it strands one per
     * computation, and a pool of any size is gone after that many. What a COMPUTATION owns is these two
     * rows; where a BLOCK carves is not the computation's to blank.
     * ⛳ THE ONE CALLER THAT MUST BLANK THEM IS THE CARVING PATH, WHICH KNOWS ITS NODE IS FRESH — and it
     * blanks them itself, in `zzpackage_start`. */
    computing_base->args[SYS__COMPUTING_BASE__BINDINGS] = 0ull;
    computing_base->args[SYS__COMPUTING_BASE__PROGRAM]  = 0ull;

    /* ⛔⛔ AND THE STATE IS NOT TOUCHED HERE, WHICH IS THE WHOLE OF WHAT KEEPS THE FILL EXCLUSIVE. The
     * state word IS the lock: `claim` takes it by swapping FREE for this block's own COMPUTING, and
     * FREE is the only value that swap accepts. Writing FREE at this point would hand the lock back in
     * the middle of the fill — and the fill is not atomic, so a second block's `claim` would succeed on
     * a base already being filled, its `init` would pass the check below on its own name, and two fills
     * would interleave: one caller's retained picture and bindings overwritten and leaked, and the base
     * left SCHEDULED for a block that was told its claim failed. Nothing afterwards could tell, because
     * the state word would read exactly as it should.
     * ⇒ ★ THE STATE SAYS WHO HOLDS THIS BASE, SO EVERY WRITE TO IT IS A HAND-OVER. There is no such
     *   thing as setting it back for tidiness — the only writes are the ones that mean to give it away,
     *   and they are the two at the bottom of this verb. */

    /* ⛔⛔ THE BINDINGS REACH IS EMPTIED HERE AND ARMED IN `begin`, AND THE REASON IS WHOSE IT IS. The
     * reach is BLOCK-LOCAL — a cache of the scopes belonging to the environment this block is reading
     * names out of — and this verb runs on whoever SCHEDULED, which is not who will read them. Filling it
     * here fills the wrong block's, and what is scheduled may not even be an environment yet: a snapshot
     * is a plain array until `begin` builds one over it, and asking it for scopes raises.
     * ⇒ ★ BLOCK-LOCAL STATE BELONGS TO THE VERB THAT RUNS ON THE BLOCK IT DESCRIBES. What this one owes
     *   the next computation is only that nothing stale is left behind. */
    sys__bindings__zzpackage_reach_forget();

    if ((program_picture == 0ull) != (bindings_viewonly == 0ull)) {
        computing_base->args[SYS__COMPUTING_BASE__PROGRAM] = SYS__COMPUTING_BASE__FAULT_HALF;
        sys__fault__raise(0ull, SYS__COMPUTING_BASE__FAULT_HALF);
        sys__computing_base__zzprivate_set_status(computing_base, SYS__COMPUTING_BASE__ERROR);
        return false;
    }
    /* ⛳ ASKED FOR NOTHING, SO THE BASE IS EMPTY AND THIS IS WHERE IT GOES BACK. The blanking shape —
     * `init(base, me, 0, 0)` — is the one caller that wants the state given up rather than passed on, so
     * FREE is written on this path and only on it. The publish is the same argument as the one at the
     * bottom: the two blanked slots must be visible before the word that invites somebody else in. */
    if (program_picture == 0ull) {
        sys__silicon__publish();
        sys__computing_base__zzprivate_set_status(computing_base, SYS__COMPUTING_BASE__FREE);
        return true;
    }

    /* ⭐⭐ A HOLD OF ITS OWN ON EACH, AND NOT A COPY OF EITHER. Both operands are things MANY bases are
     * given at once — one environment's base underneath every scope, one picture of a program handed to
     * every block that will run it — so what a base needs of them is to say "this may not go while I have
     * it". That is a count, not a chunk.
     * ⛔ AND IT IS A RETAIN RATHER THAN A TRANSFER FOR THE SAME REASON: a hold that MOVED could only ever
     * feed the first base of a fan-out. The caller keeps the one it arrived with.
     * ⛳ THE WHOLE OF WHAT A CONSUMER WRITES ON THIS PATH IS THESE TWO INCREMENTS — one word each, from
     * however many blocks are being scheduled. It is the only contention left on the way in, which is why
     * it is worth naming: everything else here is a read or a store nobody else can see. */
    (void)sys__heap_object__retain(program_picture);
    (void)sys__heap_object__retain(bindings_viewonly);

    /* ⛳ AND THERE IS NO ROLLBACK BETWEEN THEM, BECAUSE A RETAIN CANNOT FAIL. Both-or-neither is promised
     * and enforced by the half-a-state refusal above, which turns away a program without bindings before
     * anything at all is taken. */

    /* Written directly, because the generic setter refuses these two — and this is the one verb that is
     * allowed to fill them, which is exactly what that refusal is protecting. */
    computing_base->args[SYS__COMPUTING_BASE__PROGRAM]  = program_picture;
    computing_base->args[SYS__COMPUTING_BASE__BINDINGS] = bindings_viewonly;

    /* ⛔⛔ AND THE FILL IS PUBLISHED BEFORE THE STATUS SAYS SO, WHICH IS THE ONE ORDERING IN THIS VERB
     * THAT IS NOT A PREFERENCE. Whoever fills a base and whoever runs it are different blocks, and the
     * two stores above are the whole of the hand-over: without this, a reader is allowed to see SCHEDULED
     * before it sees the program, and it would begin on whatever the slots held a moment ago — a base
     * that looks filled and is not. ⇒ ★ THE FLAG MUST BE THE LAST THING ANYBODY CAN SEE, and saying so
     * is the only way to make it true. */
    sys__silicon__publish();

    /* Filled, and nobody has started: written plainly rather than swapped, because nothing that could act
     * on it is looking yet — a base reaches here held by its filler, and the store below is what ends
     * that. */
    sys__computing_base__zzprivate_set_status(computing_base, SYS__COMPUTING_BASE__SCHEDULED);
    return true;
}

/* Hand a freshly carved node to this block as its base. The one caller is the heap, at the moment it has
 * placed a base and tagged it and has nothing else to do with it.
 *
 * ⭐ IT EXISTS SO THAT SETTING A STATE PLAINLY DOES NOT HAVE TO BE ON THE PACKAGE SURFACE. What the heap
 * needs is two statements — say this block holds it, then fill it with nothing — and both are this file's
 * business. Exporting them as one named operation costs the same and closes a door: a raw state store
 * reaches ANY state from ANY state, and there is no call site that wants that.
 *
 * ⛳ IT IS THE SAME TRADE `zzpackage_get_heap_provider` MAKES: a specific verb whose name is the contract,
 * so the generic state store stays file-private.
 *
 * ⛔ IT WRITES THE STATE WITHOUT SWAPPING, WHICH IS SAFE BECAUSE nothing has
 * been told where this base is. It is not published until the heap writes its offset into the per-block
 * array, and that is the last thing the heap does. */
static __device__ inline bool sys__computing_base__zzpackage_start(sys__heap_node* computing_base,
                                                                   uint32_t block_id) {
    /* ⛔ THE PROVIDERS ARE BLANKED HERE AND NOWHERE ELSE. This node was carved a moment ago and the heap
     * allocation pool does not zero what it hands out, so both words hold whatever the last tenant left —
     * reading one of those as a chunk this block owns would carve into somebody else's room. `init` does
     * not do it, because for every other caller these two name live chunks worth keeping. */
    computing_base->args[SYS__COMPUTING_BASE__OBJECTS] = 0ull;
    computing_base->args[SYS__COMPUTING_BASE__ARRAYS]  = 0ull;
    sys__computing_base__zzprivate_set_status(computing_base,
                                              SYS__COMPUTING_BASE__COMPUTING + block_id);
    return sys__computing_base__init(computing_base, block_id, 0ull, 0ull);
}

/* ── COLLECTING WHAT IT PRODUCED, WHICH IS ALSO HOW IT IS PUT AWAY ──────────────────────────────────
 * ⚖ ARCHITECT: *"the requestor once he reads the value ... maybe clear is the wrong name, we need
 * read_result that clears everything and passes you the result sitting in the list pointer. whether it is
 * the caller or a scheduler it does not mind, he is now in charge of that reference and he either hands it
 * over to someone else or disposes of it."*
 *
 * ⭐ ONE VERB, BECAUSE READING THE RESULT AND PUTTING THE BASE AWAY ARE ONE ACT. A base whose result has
 * been taken holds nothing anybody wants, and one that has been emptied has no result to give — there is
 * no ordering of two verbs that is not either a leak or a base nobody can use again. So they are not two.
 *
 * ⛔⛔ AND MERGING THEM SHOWED THAT READING NEEDS THE SAME EXCLUSION FILLING DOES. Two collectors both
 * reading a finished base would both be handed the same reference and both believe they own it — one hold,
 * two owners, and the second release is on memory the heap allocation pool has given away. So this SWAPS out of the
 * terminal state first: exactly one caller leaves with the result, and everybody else is handed zero. That
 * is what the block id is for, and it is why "it does not mind who" is true of the OUTCOME and not of the
 * mechanism.
 *
 * ⛳ TWO SWAPS BECAUSE THERE ARE TWO ENDINGS. A compare-and-swap matches one value, and a computation
 * finishes at OK or at ERROR, so it tries both. The second only runs when the first did not match.
 *
 * ⛳ WHAT COMES BACK IS A PRE-RETAINED OBJECT, NOT A COPY.
 * The base stops naming it and the caller starts, so the reference count doesn't change.
 * The caller is in charge of it from here and either hands it on or lets it go — and if it lets it go and
 * it was the last, the heap takes the object apart, which is the whole of what disposal means.
 * ⛔ THE BINDINGS ARE RELEASED RATHER THAN HANDED OVER, because they were never this computation's to
 * give: it was lent a view of somebody's immutable structure and it is giving that view back.
 *
 * ⚠ ZERO MEANS "NOTHING FOR YOU", AND IT MEANS IT IN THREE DIFFERENT WAYS. It is not finished; somebody
 * else collected it first; or the base is at ERROR and there was never an object to hand over. None of
 * the three is an outcome of the computation, which is why none of them is distinguishable here and why
 * `status` is what a caller asks first.
 * ⭐ A COMPUTATION THAT WENT WRONG IS NOT ONE OF THEM. An error is a VALUE, so a program that evaluated
 * to one ended at OK and its error comes back through this door like any other result. What ERROR means
 * is that the MACHINE is unsound, and `sys__computing_base__fault` is where that reason is read —
 * without taking anything, and before collecting, because collecting is only how the base gets to FREE.
 *
 * ── AND `status` IS THE CATCH, WHICH IS WHY IT TAKES NOTHING AND CHANGES NOTHING ─────────────────────
 * ⚖ ARCHITECT: *"checking the status (which bypasses the lock and the clear) tells us if we need to treat
 * it as an exception or not, it is effectively held as a try-catch."*
 *
 *     if (sys__computing_base__completed(base)) {          // did it stop?
 *         const uint64_t why = sys__computing_base__fault(base);   // nonzero = UNSOUND
 *         const sys__heap_node v =                                 // a KIND and a word, not an offset
 *             sys__computing_base__read_result(base, me);          // ...and take it, once
 *     }
 *
 * ⛔ AND THE CATCH IS NOT THE STATUS, WHICH IS THE ONE THING A READER OF THIS BLOCK MUST NOT
 * CARRY AWAY WRONG. A program that threw something its own code handles evaluates to an error VALUE and
 * the base says OK — so an ordinary `try/catch` reads the KIND of what `read_result` hands back. The
 * status answers a different and much worse question: whether the machine is still sound. A nonzero
 * `fault` means scopes are half-left and objects are held by nobody, and the only move is to tear the
 * engine down. ⇒ `v` COMES BACK AS NOTHING IN THAT CASE, because there is nothing to collect.
 *
 * Asking costs one aligned read, takes no lock, and puts nothing away — so a caller may ask as often as it
 * likes and decide whether it is collecting a value or handling an exception BEFORE it commits to the one
 * collection anybody gets. It has to be asked before, not after: afterwards the base is free and says
 * nothing about what it was.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── PUTTING A FINISHED COMPUTATION AWAY, ONCE THE BASE IS ALREADY HELD ──────────────────────────────
 * The tail of collecting, with the taking removed. Everything below runs on a base that is ALREADY at
 * COMPUTING + the caller, so it takes no state and spends no permission — the caller arrived with both.
 *
 * ⭐⭐ IT IS A FUNCTION BECAUSE IT HAS TWO CALLERS AND THE SEQUENCE IS DELICATE. A scheduler
 *   collecting a result and a block putting away one the HOST already took do exactly the same five
 *   things in exactly the same order, and differ only in who took the state and whether anybody wanted
 *   the answer. Written twice, the second copy is where the reach-forget or the bindings release goes
 *   missing, and neither omission shows up as anything but a slow leak.
 *
 * ⛳ AND `threw` DECIDES ONLY WHAT IS HANDED BACK, NEVER WHAT IS PUT AWAY. The slots are emptied either
 *   way and nothing here releases what was in PROGRAM, so a caller that discards the answer is not
 *   dropping a hold — it is leaving one with whoever else already has it. */
static __device__ inline sys__heap_node sys__computing_base__zzprivate_drain(sys__heap_node* computing_base,
                                                                            uint32_t block_id, bool threw) {
    /* ⛔⛔ AT ERROR THE SLOT IS A FAULT CODE AND NOT A REFERENCE, so nothing is handed back: a caller
     * that took a code for a value would be holding a number shaped like an address. The code is read
     * with `fault` BEFORE collecting, which is what the status is for; collecting is only how the base
     * gets back to FREE. */
    /* ⛔ AND THE REACH GOES BEFORE ANYTHING IS LET GO OF. What it remembers belongs to the computation
     * being closed, and the bindings it points into are released a few lines down — so leaving it would
     * leave a base into an object that is being taken apart, for the next computation to match on. */
    sys__bindings__zzpackage_reach_forget();

    sys__heap_node result = sys__heap_node__nothing();
    if (!threw) {
        result.dtype   = (sys__kind)computing_base->args[SYS__COMPUTING_BASE__RESULT_KIND];
        result.args[0] = computing_base->args[SYS__COMPUTING_BASE__PROGRAM];
    }
    const uint64_t bindings = computing_base->args[SYS__COMPUTING_BASE__BINDINGS];
    /* ⛔⛔ EMPTIED, AND THAT IS THE WHOLE OF WHAT "TRANSFERS OWNERSHIP" MEANS HERE. The hold the base was
     * keeping is now the caller's, and it is the caller's because nothing here gives it up — so if the
     * slot kept naming the same thing there would be two holders recorded as one. ⚖ ARCHITECT: *"the
     * method retrieval should set the execution list pointer to null so there is only one holder of that
     * value."* It is also what makes a second collector harmless: the base is FREE by then and answers
     * nothing, rather than handing the same value to somebody else. */
    computing_base->args[SYS__COMPUTING_BASE__PROGRAM]     = 0ull;
    computing_base->args[SYS__COMPUTING_BASE__RESULT_KIND] = 0ull;
    if (bindings != 0ull) (void)sys__heap_object__release(bindings);

    /* Held by this caller, so `init` will fill it — and asking it for nothing is how a base is emptied.
     *
     * ⛔⛔ AND THE TWO PROVIDERS ARE LEFT ALONE, BECAUSE A BLOCK DOES NOT STOP CARVING WHEN A COMPUTATION
     * ENDS. `init` blanks the two REFERENCES and touches nothing else, which is what this caller needs:
     * the providers name chunks this block still owns and is still allocating into. Forgetting them does
     * not give them back — they stay marked as somebody's and the next allocation claims fresh ones, so a
     * block that ran a program and collected it strands its arena, every time, quietly.
     * ⛳ THE ONE CALLER THAT MUST BLANK THEM DOES IT ITSELF: `zzpackage_start` is handed a freshly carved
     * node whose words are whatever the last tenant left, so it clears both before it asks `init` for
     * nothing. That is the only place a stale provider is cleared, and it is deliberately not here —
     * anyone moving it back into `init` puts the stranding above on every collection.
     * ⚖ ARCHITECT, and this is the interval the header states: *"you then need to remove it when you get
     * a new chunk, not when you exit from compute otherwise you lock the release of that heap chunk."*
     * ⇒ The blanking is about what a COMPUTATION owns. Where a block carves is not that, and `init` is
     * where that distinction is made, so nothing is needed here beyond asking it for nothing. */
    (void)sys__computing_base__init(computing_base, block_id, 0ull, 0ull);
    return result;
}

static __device__ inline sys__heap_node sys__computing_base__read_result(sys__heap_node* computing_base,
                                                                        uint32_t block_id) {
    uint32_t was = SYS__COMPUTING_BASE__OK;
    if (!sys__computing_base__zzprivate_advance(computing_base, SYS__COMPUTING_BASE__OK,
                                      SYS__COMPUTING_BASE__COMPUTING + block_id)) {
        was = SYS__COMPUTING_BASE__ERROR;
        if (!sys__computing_base__zzprivate_advance(computing_base, SYS__COMPUTING_BASE__ERROR,
                                          SYS__COMPUTING_BASE__COMPUTING + block_id))
            return sys__heap_node__nothing();
    }
    return sys__computing_base__zzprivate_drain(computing_base, block_id,
                                                was == SYS__COMPUTING_BASE__ERROR);
}

/* ── AND THE SAME ENDING WHEN THE COLLECTOR IS THE HOST'S C DOOR ─────────────────────────────────────
 * A scheduler on another block takes a finished base with `read_result` and everything is closed inside
 * one call. The host's door cannot: reading the answer is a copy it makes for itself, and the putting
 * away is reference counting, which runs `sys` — and that door runs nothing of `sys` while the runners
 * work (▶ `eng_abi_result_release`). So the two halves come apart, and these two words are the seam.
 *
 * ⭐⭐ THE HAND-BACK IS `COMPUTING + THE OWNER`, AND NO NEW STATE WAS ADDED TO CARRY IT. There was no room
 *   for one — the status is FREE, SCHEDULED, OK, ERROR and then COMPUTING plus a block number, so every
 *   value below four is spoken for and every value above is somebody's identity. What this needed turned
 *   out not to be a new state but a state READ FROM A PLACE IT COULD NOT OTHERWISE OCCUR:
 *
 *     at the top of block N's own waiting loop, COMPUTING + N can only have been written by somebody else
 *
 *   because every route that legitimately reaches it has block N somewhere other than here — claiming
 *   (it is not scheduling), beginning (it would be evaluating), collecting (it would be in the call that
 *   collects). A scheduler on another block writes COMPUTING + ITSELF and is never mistaken for this.
 * ⇒ ★ A STATE CAN BE UNAMBIGUOUS BECAUSE OF WHERE IT IS READ RATHER THAN WHAT IT IS, and that is worth
 *   the paragraph it costs, because nothing about the value says so.
 *
 * ⛔⛔ IT READS AND MUST NOT WRITE, WHICH IS THE WHOLE REASON THE SEAM GREW A WORD FOR IT. The obvious
 *   spellings — adding zero, or swapping a value for itself — both put the word back, and putting back
 *   what you read is how you destroy what somebody else wrote in between. `MEASURED` on the device path
 *   and NOT re-derived on the host: with the waiter compare-and-swapping this word, a host write of
 *   SCHEDULED never survived, while the two operands written beside it on the same node always did. The
 *   rule stands either way. `status` stays a plain read: every OTHER caller of it
 *   is reading a word it wrote itself.
 *
 * ⛳ AND THE ANSWER IT PUTS AWAY IS NOT LOST, IT IS ALREADY SOMEWHERE ELSE. The host copied the value out
 *   before handing the base back, and the drain empties the slot without releasing — so the hold the
 *   base was keeping becomes the host's, by the same rule that makes `read_result` a transfer. */
static __device__ inline uint32_t sys__computing_base__zzengine_peek_status(const sys__heap_node* computing_base) {
    if (!sys__computing_base__is(computing_base)) return SYS__COMPUTING_BASE__FREE;
    const unsigned int* word =
        (const unsigned int*)&computing_base->args[SYS__COMPUTING_BASE__STATUS];
    return (uint32_t)sys__silicon__read_u32(word);
}

/* Put away a computation whose answer somebody else already has. The base is held by this block because
 * whoever took the answer said so, so there is no state to win and nothing to wait for.
 * ⛳ `threw` IS FALSE AND THE CHOICE DOES NOT MATTER, which is the point of it deciding only what is
 * handed back: the answer is discarded here either way, and a fault code in the slot is emptied rather
 * than released exactly as a value would be. */
static __device__ inline bool sys__computing_base__zzengine_release_result(sys__heap_node* computing_base,
                                                                          uint32_t block_id) {
    if (sys__computing_base__zzengine_peek_status(computing_base)
            != SYS__COMPUTING_BASE__COMPUTING + block_id)
        return false;
    (void)sys__computing_base__zzprivate_drain(computing_base, block_id, false);
    return true;
}

/* ── FINDING THE ONE THIS BLOCK IS WORKING WITH ──────────────────────────────────────────────────────
 * ⚖ ARCHITECT: the system register carries a `computing_base_array` indexed on the block id, so one
 * reference reaches every block's base and the lisp has an access point it can name.
 *
 * ⛳ IT IS WIRED, AND IT IS THE LIVE PATH. `system_register__impl.cuh` holds the register; the boot stands
 * it up (`eng__boot__publish_register`) and writes row `SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY` with
 * the one allocation that holds every block's base end to end; a program reaches rows of it by name,
 * through `sys__system_register__get` and `__set`. So the lookup below RUNS: it goes through
 * `sys__heap__zzpackage_computing_base`, which reads that row and adds the block id.
 *
 * ⛳ IT TAKES THE BLOCK ID RATHER THAN READING IT. A block asking for its own is the common call and not
 * the only one: a teardown walking every block wants each in turn, and a function that reads the id for
 * itself cannot be asked for anyone else's.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node* sys__computing_base__get(uint32_t block_id) {
    /* THE SHAPE, AND IT IS BUILT: the body is `sys__heap__zzpackage_computing_base`, one call away, and
     * this is a forward to it rather than a wrapper with anything of its own to say. What it does there:
     *
     *   sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
     *   uint64_t       at    = sys__heap__object_full_address(where.args[0] + block_id)->args[0];
     *   return sys__heap__object_full_address(at);
     *
     * ⚠ AND THE HOLD A ROW READ WOULD OTHERWISE OWE IS NOT OWED, WHICH IS SETTLED IN TWO PLACES AND IN
     * BOTH OF THEM DELIBERATELY. The row is PLACED as a plain number rather than as a counted reference —
     * `sys__system_register__publish` argues it, and the argument is this path: a row the allocator reads
     * on the way to every allocation would otherwise carry two count updates. And the base's own slot is
     * read with `sys__heap__object_full_address` rather than through the array's `get`, for the same
     * reason one step down — a `get` hands out a hold, so it would put a retain and a release either side
     * of an addition. Both sites say so where they do it, so neither is a silent shortcut.
     *
     * ⛳ A PLAIN INDEXED ARRAY, NOT A LIST, AND THAT IS WHAT THE ADDITION ABOVE IS. ⚖ ARCHITECT: *"it
     * needs to be indexed by the block id, and it is a init setup so i have no worry about
     * overallocation ... or things that might happen to declare an array that grows outside the heap
     * guardrails."* Every reason a list exists — growth, insertion, an extent nobody knows yet — is absent
     * here: one slot per block, sized once at boot, never resized. A list would put a structure with a
     * lifetime of its own into the boot path to solve problems the boot path does not have.
     *
     * ⛳ AND NOTHING HERE IS PENDING. The register is stood up by the boot, its first row is written with
     * the one allocation holding every block's base end to end, and this lookup is what the allocator
     * itself runs on the way to work. ⚠ WHAT IS UNMEASURED IS THE COST of the row read — whether it is
     * still the scalar, hoisted load the heap's argument assumes. The file holding the body says so beside
     * it, and what would decide it is a program worth timing rather than a spill count. */
    return sys__heap__zzpackage_computing_base(block_id);
}

/* ── AND WHAT HAPPENS WHEN A RELEASE REACHES ONE, WHICH IS THAT NOTHING SHOULD ────────────────────────
 * A computing base outlives everything it names. It is placed once when the block starts and is never
 * made, so it has nothing to hand over and no room to give up — and giving its room up would put the
 * chunk the block carves from back in the heap allocation pool while the block is still running.
 *
 * ⇒ So this is not the quiet arm it looks like. Reaching it means something counted a base as an
 * ordinary object, and raising is what makes that visible at the moment it happens rather than as a
 * block that stops making sense some time later.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__computing_base__zzpackage_never_release_internal(sys__heap_node* head,
                                                                                   uint64_t* releaser_stack) {
    (void)releaser_stack;
    sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NEVER_ENDS);
    /* ⛔ AND IT PUTS THE COUNT BACK, WHICH IS NOT TIDINESS. Reaching zero is what got us here and it was
     * WRONG — this thing outlives everything that could name it — so leaving the zero would let the room
     * go back and hand the heap allocation pool the block's own chunk, which is far worse than the mistake that caused
     * it. Writing one says the true thing: it is not dead. Nothing else needs to know it is special. */
    if (head != 0) SYS__HEAP_OBJECT__COUNT(head) = 1u;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__IMPL_CUH */
