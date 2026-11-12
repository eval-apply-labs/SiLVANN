#ifndef SILVANN__PACKAGES_SYS_CPU_SCHEDULER__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_SCHEDULER__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap_node__header.cuh"                 /* the node a value is */
#include "../contracts/objects/scheduler.cuh"    /* a request's and a response's cells, the holder, the fault words */

/* ══ THE SCHEDULER — A QUEUE IN FRONT OF EVERY BLOCK, AN ANSWER FOR EVERY REQUEST ═══════════════════════
 *
 * ⚖ ARCHITECT 2026-11-03: *"add a input list ... so the gpu0 adds to the scheduler his request ... then the cpu gemv
 * engine will eventually get to their request and execute it"* · 2026-11-05: *"it becomes (sys_compute cu bindings
 * function result), where result is a heap object that contains the result of the computation (and it starts at
 * waiting) so we dissolve the whole issue"*.
 *
 * A program's work goes on a block's QUEUE, which any number of blocks append to at once, and its value goes into the
 * ANSWER the request carries (▶ `answer.cuh`) — made before the work is asked for, so nothing has to find the value
 * afterwards, and whoever holds the answer reads it:
 *
 *     (sys__scheduler__add cu names program)         queue it on block `cu`: a new answer, handed back
 *     (sys__scheduler__add cu names program answer)  the same, into `answer`, which is handed back
 *     (sys__result answer)                           #t once the value is there, #f while it is on its way
 *     (sys__wait_for_result answer)                  the value, once it is there — the computation parked on the
 *                                                    answer meanwhile, its block free; where it may not move (the
 *                                                    host's, inline), waited for where it stands, bounded
 *     (sys__wait_for_device cu)                      #t once block `cu` is idle — its base free, its queue empty,
 *                                                    nothing parked — a probe of the device, no value handed over
 *     (sys__scheduler__continue cu names program)    this computation goes on as `program` on block `cu` in `names` (0:
 *                                                    the names it runs in), into the same answer; this step fills
 *                                                    nothing, the next does
 *     (sys__scheduler__continue cu names program '(ra rb))
 *                                                    the same, run once the answers the names `ra` and `rb` mean here
 *                                                    are filled — `ra` and `rb` bound to what they came to in its
 *                                                    environment
 *     (sys__scheduler__pid)                          the process this computation is of: the asker's, inherited by
 *                                                    what it asks and what it goes on as
 *     (sys__scheduler__kill pid)                     its requests off every queue, their answers SCKL; its steps
 *                                                    still running ended at their next move or slice
 *     (sys__scheduler__compute_inline 'form)         `form`'s value, the form run with this computation never given
 *                                                    up at a slice; after it, preempted as it was before
 *     (sys__scheduler__preemptive)                   #t, or #f inside `compute_inline`: whether a spent slice may
 *                                                    give this computation's block up
 *     (sys__scheduler__continue_on cu)               this computation moves to block `cu` — its environment, and the
 *                                                    program as it stands — and goes on there from the next form,
 *                                                    into the same answer; nothing of it runs here after
 *
 * ⚖ 2026-11-05 *"sys__compute should specify a new closure to compute against ... (sys__scheduler__add CU names expert
 * result) repeated with as much as the async results are needed and the result variables become inserted in the
 * callback values"* · *"sys__scheduler__add returns an object, which we can bind with let ... (let ((ra
 * (sys__scheduler_add ...)) (rb (sys__scheduler_add ...)) (sys__scheduler__continue card names second_half '(ra rb) ))
 * and names also contains the references to ra and rb as bindings"*.
 * ⭐ A COMPUTATION MAY CONTINUE (⚖ 2026-11-04 *"simply a self-queuing exercise"*): the next step queued on any block,
 * the block it runs on among them — taken when this one ends, behind whatever was queued before it, so two
 * computations on one block interleave by the queue alone. The step that does not continue fills the answer.
 * ⭐ AND A STEP MAY WAIT FOR ANSWERS WITHOUT HOLDING ITS BLOCK (⚖ 2026-11-05 *"if we are waiting on a computation and
 * sys_result still says waiting we can shove the computation at the back of the current thread computation queue and
 * make the other progress"*): continued with the names of answers, it runs once every one of them is filled, in a new
 * closure over its names with those names bound to the values they came to. A block that takes it earlier parks it and runs what is
 * queued behind it; the block that fills the last answer rings it. So two computations on one card each ask the CPU
 * and go on, and each one's next step runs while the other's experts are computed.
 * ⭐ WHAT A COMPUTATION CAME TO IS READ FROM ITS ANSWER, NEVER FROM THE BLOCK THAT RAN IT (⚖ 2026-11-11 *"no one names a
 * worker to read a computing variable, they just name the return variable"*): a block may hold several askers' work, a
 * pipeline's among them, so a block names no result. `(sys__wait_for_device cu)` is the block's own state, idle or not.
 *
 * ⛳ A CLOSURE RUN HERE AND NOW, IN THE MIDDLE OF A COMPUTATION, IS NOT THERE YET: the evaluator runs one evaluation a
 * block at a time and nothing inside it may start another (▶ `engine/eval.cuh`), so a closure is run as a block's next
 * computation — which is what a continued step is.
 * WHO RUNS IT. The block itself, between computations: an idle block whose base is free takes the oldest request
 * off its own queue, claims its own base in the scheduler's name (`SYS__SCHEDULER__HOLDER`) with the answer as its
 * answer word, and runs it as any computation runs (▶ `engine/opcodes.cuh`). At its end the base files its value
 * into the answer and is free again. Every request comes this way: a program's, and the host's through the engine's
 * doors (`dispatch`, `place`), the host carving from a base of its own in the slot past the blocks
 * (▶ `sys__computing_base__host`). So requests are served in the order they arrived, from every caller, and nothing
 * but a block's own runner touches its base.
 * WHERE THEY LIVE. One row of the register, an array a slot a block and one for the host, a list in each slot, stood
 * up once by sys's init and never written again — so finding a block's queue reads two words and locks nothing; the
 * lists themselves are shared and reached only through `sys__list__shared_*`, and never by a program: a request is
 * three cells on a queue, put on and taken off whole under the list's lock (`_append_cells`, `_take_cells`), so
 * nothing is made for it on one block and let go on another.
 * WHAT IT DOES NOT DO. Choose: a request names its block. Serve a block that is running a program rather than parked —
 * a block takes nothing from its queue until it parks, so `sys__scheduler__add` refuses the block it runs on.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The queues, one each a block, made and written into the register. sys's init, once the computing bases stand; it
 * answers false having raised. */
static __device__ SYS__NOINLINE bool sys__scheduler__zzpackage_stand_up(void);

/* A block's queue: the list, borrowed — perennial for the heap's life — or 0. */
static __device__ inline uint64_t sys__scheduler__queue(uint32_t block);

/* ⭐ A HOLD LET GO OF AT HOME (NN-181, ⚖ 2026-11-11 *"the deallocation just executes the deallocation code, which by
 *   itself will ask a decommission on the rightful owner"*): an object whose memory is one worker's — a buffer carved
 *   from its pool, filed back into its free list, which only its own thread writes — may lose its last hold on another
 *   worker's thread. Its release hook then hands a hold of it to block `cu`, the worker that owns it, and returns: the
 *   list takes its own hold, so the object outlives the release. That block's runner lets go of it between its
 *   computations (`zzengine_decommissions`), on its own thread, where the same release files it home. Answers whether
 *   it was handed over — false where the scheduler is not up, and the caller does what it did before. */
static __device__ inline bool sys__scheduler__decommission_on(uint32_t cu, uint64_t object);
/* What block `me` was handed to let go of, let go of here: answers how many. Its runner's, between computations. */
static __device__ inline uint32_t sys__scheduler__zzengine_decommissions(uint32_t me);

/* Queue a request on block `cu`, its value for `answer`, armed here (▶ `sys__answer__zzpackage_arm`). The request takes
 * its own holds on the bindings, the picture and the answer; the caller keeps its. Answers whether it was queued,
 * false having raised — the answer left as it was found where it was refused busy, and EMPTY where it was armed and
 * could not be queued. */
static __device__ inline bool sys__scheduler__add(uint32_t cu, uint64_t bindings, uint64_t program, uint64_t answer);

/* The computation on `base` goes on as `program` on block `cu` in `names` — 0 for the names it runs in — into the
 * same answer, run once every answer `gate` names is filled, where it is not 0: an array of pairs, a name and the
 * answer it means; the computation's own value is dropped. Answers whether it went on — false where `base` has no
 * answer to fill, its step went on already, or `gate` names an answer nobody asked into. */
static __device__ inline bool sys__scheduler__continue(sys__heap_node* base, uint32_t cu, uint64_t names, uint64_t program,
                                                       uint64_t gate);

/* The computation on `base` moves to block `cu`: its environment, the program as it stands and where its evaluation
 * stands, into the same answer. This says where, and the evaluator, finding it after the verb (`zzengine_handed`: the
 * block plus one, or 0, the word cleared), hands over what it holds (`zzengine_move`) and stops; the block that takes
 * it runs the evaluation on from the same form (`zzengine_resumed`: the places and the form, the places' hold handed
 * over). Its collection mode and how many `compute_inline` it stands inside go with it. False, having raised, where
 * `base` has no answer to fill or its step went on already (SCNQ), or where another program shares its environment
 * (SCSH: the host's, or one a continuation shares). */
static __device__ inline bool     sys__scheduler__continue_on(sys__heap_node* base, uint32_t cu);
static __device__ inline uint64_t sys__scheduler__zzengine_handed(void);
static __device__ inline bool     sys__scheduler__zzengine_move(sys__heap_node* base, uint32_t cu, uint64_t program,
                                                                uint64_t places, uint64_t current);
static __device__ inline bool     sys__scheduler__zzengine_resumed(uint64_t* places, uint64_t* current);

/* The slice (▶ "THE SLICE" in `scheduler__impl.cuh`): how many cycles a computation runs before it gives its block up to
 * what waits there — 0 for never — set by sys's host init from `boot_commands`' `sys__scheduler__slice_cycles`, or
 * `sys__scheduler__slice_us` in the clock's ticks of that many microseconds; the clock it is counted on; and whether the
 * computation on `base` should give its block up now its slice is spent — something waits there, and it may move. */
static __device__ inline void     sys__scheduler__zzpackage_set_slice(uint64_t cycles);
static inline uint64_t            sys__scheduler__zzpackage_ticks_of_us(uint64_t us);
static __device__ inline uint64_t sys__scheduler__zzengine_slice(void);
static __device__ inline uint64_t sys__scheduler__zzengine_clock(void);
static __device__ inline bool     sys__scheduler__zzengine_should_yield(sys__heap_node* base);

/* The processes (▶ "THE PROCESSES" in `scheduler__impl.cuh`): the process of the step this runner runs (0 where none
 * runs); a process killed — its requests taken off every queue, their answers SCKL, and remembered, so a block refuses
 * one taken later and a step still running ends at its next move or slice — answering how many it took off; whether one
 * is killed; and for the evaluator, after a move refused or a slice spent, whether the running step is of one killed —
 * its answer filled with SCKL then, and the step to end. */
static __device__ inline uint64_t sys__scheduler__pid(void);
static __device__ inline uint64_t sys__scheduler__kill(uint64_t pid);
static __device__ inline bool     sys__scheduler__zzpackage_killed(uint64_t pid);
static __device__ inline bool     sys__scheduler__zzengine_killed(sys__heap_node* base);

/* A process for the host, and one asked to stop (NN-160, ▶ "A PROCESS FOR THE HOST" in `scheduler__impl.cuh`): a new
 * process, from the counter every process is made from; this thread's requests made as process `pid` from here on (0:
 * each a new one, as before), answering the one they were made as; a process asked to stop at the token it is on —
 * remembered, its program reading it — and whether the running step's process has been, or killed. */
static __device__ inline uint64_t sys__scheduler__process_new(void);
static __device__ inline uint64_t sys__scheduler__process_as(uint64_t pid);
static __device__ inline bool     sys__scheduler__stop(uint64_t pid);
static __device__ inline bool     sys__scheduler__stopped(void);

/* The inline sections (▶ "INLINE" in `scheduler__impl.cuh`): one entered and one left, by `compute_inline`; and whether
 * the computation this runner runs may be given up at a slice — not inside one. */
/* A wait parked (▶ "A WAIT PARKED" in `scheduler__impl.cuh`): whether the computation on `base` may be parked on an
 * answer not there yet — one with an answer to fill and an environment of its own, not inline, not refused a park in
 * this step — and the hand-off that parks it, on its own block with `answer` its gate. */
static __device__ inline bool     sys__scheduler__zzpackage_may_park(sys__heap_node* base);
static __device__ inline void     sys__scheduler__zzpackage_park_on(uint64_t answer);

static __device__ inline void     sys__scheduler__zzpackage_inline_enter(void);
static __device__ inline void     sys__scheduler__zzpackage_inline_leave(void);
static __device__ inline bool     sys__scheduler__preemptive(void);

/* The gate of the step block `me` is running, BORROWED — the block holds it until the step is put away — or 0.
 * `zzengine_bind_gate`: its names bound to its answers' values in the step's environment (`enter`), once `begin` has
 * built it, or those scopes left again once its evaluation ends. `zzengine_parked`: how many steps block `me` has parked, each waiting for its gate. Only
 * block `me`'s runner writes any of them. */
static __device__ inline uint64_t sys__scheduler__zzpackage_gate(uint32_t me);
static __device__ inline void     sys__scheduler__zzengine_bind_gate(sys__heap_node* base, uint32_t me, bool enter);
static __device__ inline uint32_t sys__scheduler__zzengine_parked(uint32_t me);

/* The block's side (▶ `engine/opcodes.cuh`). `next`: its base, free, filled in the scheduler's name from the oldest of
 * its parked steps whose gate filled, or else from the oldest request on its queue that may run — one waiting for a
 * gate parked on the way — the request's answer as its answer word; answers whether there was one. `post`: a value of
 * the base's scheduled computation filed into its answer — answers whether the base has an answer word (a step that
 * continued fills nothing and is put away all the same), having raised if the filing failed. `put_away`: a finished
 * scheduled base given back free, its value's hold and its answer's let go of, the answer word cleared first.
 * `post_error`: a scheduled base at ERROR — it could not begin — answered as an error value and put away; answers
 * whether it was one.
 * ⛔ THE VALUE IS FILED BEFORE THE BASE FINISHES, while this block holds it at COMPUTING: the answer is ready before
 * the base says it is done. */
static __device__ inline bool sys__scheduler__zzengine_next(sys__heap_node* base, uint32_t me);
static __device__ inline bool sys__scheduler__zzengine_post(sys__heap_node* base, uint32_t me, sys__heap_node value);
static __device__ inline void sys__scheduler__zzengine_put_away(sys__heap_node* base, uint32_t me);
static __device__ inline bool sys__scheduler__zzengine_post_error(sys__heap_node* base, uint32_t me);
/* `doze`: block `me`, nothing to do for a while, asleep on its door until a request is queued for it or `ns` pass —
 * the engine's await loop's wait once quiet; every request queued rings its block's door. `ring_all`: every block's
 * door rung, for the shutdown's stop word. */
static __device__ inline void sys__scheduler__zzengine_doze(uint32_t me, uint64_t ns);
static __device__ inline void sys__scheduler__zzengine_ring_all(uint32_t blocks);

#endif /* SILVANN__PACKAGES_SYS_CPU_SCHEDULER__HEADER_CUH */
