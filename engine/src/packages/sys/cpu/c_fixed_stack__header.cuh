#ifndef SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/defaults.cuh"
#include "../contracts/objects/c_fixed_stack.cuh" /* its constants, fault words and layout */
/* ══ THE C FIXED STACK — AN ARRAY OF NUMBERS AND HOW MANY ARE IN IT ═════════════════════════════════
 *
 * An array of `capacity` numbers and an index. A push writes its number at the index and raises the
 * index; a pop lowers the index and answers the number there. The last number pushed is the first one
 * popped. It holds `uint32_t`, and what a number means belongs to whoever fills it.
 *
 * ⚖ ARCHITECT 2026-11-02: *"since we know how many heap chunks we have it can simply be an array of
 * max_len with an index, everytime we push the index goes up and we write the value and when we pop the
 * index goes down and we return the value ... call it c_fixed_stack to make sure it is not confused for
 * a lisp verb"*.
 *
 * ⛔ IT IS C ONLY, AND IT IS NOT A HEAP OBJECT. Its room comes from the platform through the silicon seam,
 * which is what lets it hold the heap's own free chunks: a stack carved from the heap (`sys__stack`, the
 * Lisp's) would claim a chunk to grow into while putting a freed one away. It has no kind and no row in
 * `sys__LANGUAGE_CONTRACT__OBJECTS`, so the Lisp teardown cannot reach one; it is held by one C caller,
 * counts no holders, and ends at `dispose`. It does not grow: it is the capacity it was made with.
 *
 * ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 * It is `create` · `dispose` · `push` · `pop` · `count`, declared at the foot of this file.
 *
 *     sys__c_fixed_stack* s = sys__c_fixed_stack__create(capacity);   the room comes with it
 *     sys__c_fixed_stack__dispose(s);                                 and goes back with it
 *
 * WHAT THE CALLER OWES
 *   a capacity      at least one, and `sizeof(sys__c_fixed_stack) + capacity × 4` bytes that fit 32 bits.
 *                   CHECKED, as SIZE and TOO_BIG, and refused rather than rounded down.
 *   big enough      the most that can ever be in it. NOT CHECKED, and it cannot be: the stack does not
 *                   know how many things its caller owns. `push` refuses past it, as FULL.
 *   one dispose     per `create`. NOT CHECKED: a stack is named by no value and counted by nothing, so a
 *                   missed `dispose` is room the platform never gets back.
 *   an empty stack  to dispose one. CHECKED, as NOT_EMPTY, and the room stays: for the heap what is in it
 *                   is chunks, and abandoning them loses those chunks with every count still balancing.
 *                   Pop until it is empty first.
 *
 * WHAT IT PROMISES BACK
 *   order           `pop` answers the last number pushed. For the heap that is the chunk freed last,
 *                   the one most likely still in the caches.
 *   no repeats      a number popped is gone until somebody pushes it again.
 *   a quiet empty   `pop` on an empty stack answers false and raises NOTHING: being empty is a state, and
 *                   a pool that has run out is its caller's news to break. THERE IS EXACTLY ONE SILENT NO;
 *                   every other false comes with a named fault word, and a stack never made raises
 *                   NO_STACK rather than reading as an empty one.
 *   a count         `top`, read as one aligned word, so it was true at the instant it was read.
 *   a dead handle   ⛔ IS NOT ONE OF THESE. `dispose` hands the whole structure back to the platform, so
 *                   a pointer kept past it dangles: until the allocator hands the room on, the cleared
 *                   witness answers NO_STACK, and after that whatever the new tenant wrote there.
 *
 * ── MORE THAN ONE RUNNER ────────────────────────────────────────────────────────────────────────────
 * The host evaluator runs a thread a card and one for the orchestrator, and each of them claims and frees
 * heap chunks, so pushes and pops meet. A push moves the index and writes a slot, a pop moves it and
 * reads one, and both work at the same end: in whichever order the two steps are done, a third runner can
 * arrive between them. The `lock` word makes them one step. A compare-and-swap takes it, another gives it
 * back, and between the two are the index and one slot. A runner that misses backs off by
 * `LOCK_BACKOFF_CYCLES` plus its block id, so two that missed together do not retry together, and gives
 * up after `LOCK_TRIES` with LOCK — the bounded wait every lock in this package has (▶
 * `sys__heap_object__lock`). The hold is a few instructions, but a holder can be descheduled or
 * interrupted inside it, and then every runner that wants the stack waits until it runs again. So the
 * bound is set past any such pause, not past the hold: a wait that reaches it is a defect
 * (▶ `contracts/defaults.cuh` for the number and what measured it).
 *
 * The word names its holder. Taking it writes the asking thread's token, a number the thread draws once
 * and no other live thread holds, and giving it back swaps that token, and only that token, back to
 * zero. It is not the block id, because two threads can serve one block id. So a thread giving back a
 * lock it does not hold raises LOCK and leaves the word naming the one that does; a thread asking again
 * for a lock it holds raises HELD at once, where it would otherwise wait out the bound on itself; and a
 * debugger looking at a stuck stack reads in the word which thread has it. Two threads serving one block
 * id back off alike, and the bound is what still ends their wait.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

static __device__ inline sys__c_fixed_stack* sys__c_fixed_stack__create (unsigned int capacity);
static __device__ inline void                sys__c_fixed_stack__dispose(sys__c_fixed_stack* stack);
static __device__ inline bool                sys__c_fixed_stack__push   (sys__c_fixed_stack* stack, uint32_t value);
static __device__ inline bool                sys__c_fixed_stack__pop    (sys__c_fixed_stack* stack, uint32_t* out);
static __device__ inline unsigned int        sys__c_fixed_stack__count  (const sys__c_fixed_stack* stack);

#endif /* SILVANN__PACKAGES_SYS_CPU_C_FIXED_STACK__HEADER_CUH */
