#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SCHEDULER_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SCHEDULER_CUH

/* This file needs nothing: every name in it is its own. ▶ `cpu/scheduler__header.cuh` */

/* Who a base is held by while its block fills it from its own queue: no block's id (block ids are dealt from zero
 * and stay far below it), so a base claimed by the scheduler says so in its status word. */
#define SYS__SCHEDULER__HOLDER 0x00FFFFFFu

/* How many blocks have a door to doze on (▶ `cpu/scheduler__impl.cuh`): a host's runners, at most. A block past them
 * naps instead. */
#define SYS__SCHEDULER__DOORS 64u

/* A request on a block's queue: these four cells, one after another. */
#define SYS__SCHEDULER__REQUEST_ANSWER   0u   /* where its value goes: an answer (▶ `cpu/answer.cuh`)  */
#define SYS__SCHEDULER__REQUEST_BINDINGS 1u   /* the read-only names, as `sys__scheduler__add` takes them   */
#define SYS__SCHEDULER__REQUEST_PROGRAM  2u   /* the program's picture                               */
#define SYS__SCHEDULER__REQUEST_GATE     3u   /* what it waits for before it may run: an array of pairs, a
                                                 name and the answer it names — or nothing                  */
#define SYS__SCHEDULER__REQUEST_CELLS    4u

/* How many steps a block keeps parked, each waiting for its gate's answer to fill (▶ `sys__scheduler__zzengine_next`).
 * One past them stays on the queue, taken again behind what came after it. */
#define SYS__SCHEDULER__PARKED 16u

/* A base whose step has continued its computation (▶ `sys__scheduler__continue`): its answer word, the answer handed to
 * the next step, so what this one came to is dropped. No answer's offset: the heap stays far below it. */
#define SYS__SCHEDULER__CONTINUED 0xFFFFFFFFFFFFFFFFull

#define SYS__SCHEDULER__FAULT_NOT_UP     0x53434E55ull   /* "SCNU" — the queues are Not Up                    */
#define SYS__SCHEDULER__FAULT_NOT_QUEUED 0x53434E51ull   /* "SCNQ" — a request Not Queued: no room, or the lock */
#define SYS__SCHEDULER__FAULT_NOT_ASKED  0x53434E41ull   /* "SCNA" — read for an answer Nobody Asked for: no
                                                            request was made into it, or of that block       */
#define SYS__SCHEDULER__FAULT_LOST       0x53434C53ull   /* "SCLS" — a computation's value that could not be
                                                            filed, so it is Lost: its answer waits for nothing */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SCHEDULER_CUH */
