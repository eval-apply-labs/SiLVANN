#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_C_FIXED_STACK_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_C_FIXED_STACK_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../defaults.cuh"

/* Turned in `contracts/defaults.cuh`, guaranteed here — see the note on the lock's pair in
 * `heap_object__header.cuh` for why both files carry a value. */
#ifndef SYS__C_FIXED_STACK__LOCK_TRIES
#define SYS__C_FIXED_STACK__LOCK_TRIES 1048576u
#endif
#ifndef SYS__C_FIXED_STACK__LOCK_BACKOFF_CYCLES
#define SYS__C_FIXED_STACK__LOCK_BACKOFF_CYCLES 10u
#endif

#define SYS__C_FIXED_STACK__FAULT_NO_STACK  0x43534E53ull  /* "CSNS" — asked of a stack never made, or disposed */
#define SYS__C_FIXED_STACK__FAULT_SIZE      0x4353535Aull  /* "CSSZ" — a capacity of zero                      */
#define SYS__C_FIXED_STACK__FAULT_TOO_BIG   0x43534247ull  /* "CSBG" — a capacity whose room cannot be counted */
#define SYS__C_FIXED_STACK__FAULT_NO_ROOM   0x4353524Dull  /* "CSRM" — the platform had no room to give        */
#define SYS__C_FIXED_STACK__FAULT_FULL      0x4353464Cull  /* "CSFL" — pushed past its capacity                */
#define SYS__C_FIXED_STACK__FAULT_LOCK      0x43534C4Bull  /* "CSLK" — its lock not had in time, or given back by a thread not holding it */
#define SYS__C_FIXED_STACK__FAULT_HELD      0x43534844ull  /* "CSHD" — its lock asked for by the thread already holding it */
#define SYS__C_FIXED_STACK__FAULT_NOT_EMPTY 0x43534E45ull  /* "CSNE" — disposed with values still in it         */

/* The first field is a witness, because the room comes from the platform and not from a pool: `dispose`,
 * `push` and `pop` tell a stack from a stranger by this value, since room from the platform has no range
 * to test an address against. It catches a pointer to something that was never a stack. It does not
 * catch one kept past `dispose`, whose bytes then belong to the allocator. */
#define SYS__C_FIXED_STACK__LIVE 0x43534C56u               /* "CSLV", written by create, cleared by dispose */

/* `top` counts what is in it: zero is empty, the last value pushed is `value[top - 1]`, and the next push
 * writes `value[top]`. `top` and the slot it names change together under `lock`. */
typedef struct {
    unsigned int live;      /* SYS__C_FIXED_STACK__LIVE while this is a stack, zero once it is not */
    unsigned int lock;      /* zero free, or the token of the thread holding it */
    unsigned int capacity;  /* the most values it holds */
    unsigned int top;       /* how many values it holds */
    uint32_t*    value;     /* `capacity` values, in the room just after this structure */
} sys__c_fixed_stack;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_C_FIXED_STACK_CUH */
