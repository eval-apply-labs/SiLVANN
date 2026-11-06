#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_ANSWER_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_ANSWER_CUH

/* This file needs nothing: every name in it is its own. ▶ `cpu/answer.cuh` */

/* An answer: a head and the node that is the value. */
#define SYS__ANSWER__NODES  2u
#define SYS__ANSWER__STATE  1u   /* args[1] of the head — args[0] is the lock in every kind: one of the three below */
#define SYS__ANSWER__WAITERS 2u  /* args[2] of the head: a bit a block, the blocks with a step parked until it fills —
                                    two 32-bit words, so a block's door is at most 64 (`SYS__SCHEDULER__DOORS`)    */

/* What an answer says. */
#define SYS__ANSWER__EMPTY    0u   /* made, and asked of nobody                         */
#define SYS__ANSWER__WAITING  1u   /* a computation is on its way to filling it          */
#define SYS__ANSWER__READY    2u   /* filled: the value is there, and stays until asked again */

#define SYS__ANSWER__FAULT_KIND 0x414E534Eull   /* "ANSN" — asked of something that is Not an answer          */
#define SYS__ANSWER__FAULT_BUSY 0x414E5342ull   /* "ANSB" — an answer asked again while its computation is still
                                                   on its way: Busy                                          */
#define SYS__ANSWER__FAULT_LATE 0x414E534Cull   /* "ANSL" — a value filed into an answer nobody was waiting on:
                                                   Late, or filed twice                                      */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_ANSWER_CUH */
