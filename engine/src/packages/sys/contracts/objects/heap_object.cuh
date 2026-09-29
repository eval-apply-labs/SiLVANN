#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_OBJECT_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_OBJECT_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../defaults.cuh"

/* ── WHAT CAN GO WRONG WITH AN OBJECT, IN ITS OWN WORDS ─────────────────────────────────────────────
 * Each cause has its own word. A log with one word for several causes tells a reader that something
 * went wrong and nothing about what, and the first four below want
 * genuinely different responses: a bad reference is a caller bug, a count below zero is an accounting bug,
 * a missing row is an unfinished kind, and a null argument is a forgotten one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__HEAP_OBJECT__FAULT_UNKNOWN_KIND    0x4F424A4Bull   /* "OBJK" — a dtype the dispatch has no row for  */
#define SYS__HEAP_OBJECT__FAULT_NOT_A_REFERENCE 0x4F424A52ull   /* "OBJR" — an offset that cannot name one of ours */
#define SYS__HEAP_OBJECT__FAULT_OVER_RELEASED   0x4F424A4Full   /* "OBJO" — given up more times than it was held */
#define SYS__HEAP_OBJECT__FAULT_NO_VALUE        0x4F424A4Eull   /* "OBJN" — a null cell or value handed in       */
#define SYS__HEAP_OBJECT__FAULT_NEVER_ENDS      0x4F424A45ull   /* "OBJE" — a kind that must not be taken apart  */
#define SYS__HEAP_OBJECT__FAULT_UNKNOWN_VERB 0x4F424A55ull  /* "OBJU" — a word no package publishes */
#define SYS__HEAP_OBJECT__FAULT_NOT_CONSTRUCTIBLE 0x4F424A42ull /* "OBJB" — a kind a program may not build */
#define SYS__HEAP_OBJECT__FAULT_NOT_CLONABLE    0x4F424A43ull   /* "OBJC" — a kind with no way to be copied      */
#define SYS__HEAP_OBJECT__FAULT_LOCK            0x4F424A4Cull   /* "OBJL" — a lock taken or given back wrongly */

#define SYS__HEAP_OBJECT__LOCK        0u      /* args[0] of an allocation head, in every kind of object */
#define SYS__HEAP_OBJECT__LOCK_FREE   0u
#define SYS__HEAP_OBJECT__LOCK_HELD   1u
/* The lock's two dials are TURNED in `contracts/defaults.cuh` and GUARANTEED here.
 *
 * ⚖ ARCHITECT: *"i can comment out the default in the variables and not set it at compile time and i
 * still would like a functioning program."* So the guards below are a floor and not a duplicate policy:
 * `contracts/defaults.cuh` is read first, and if it has a value these do nothing. If somebody comments one
 * out to see what happens, what happens is that the code still builds and runs.
 * ⛳ THIS IS THE LEVER RULE ONE TIER UP, APPLIED TO A DIAL: an override is a thing you must be able to
 * DELETE, and deleting one must not leave nothing where a value is required.
 * ⚠ THE PRICE IS TWO COPIES OF EACH NUMBER, WHICH CAN DISAGREE — said plainly because nothing checks it:
 * by the time this file is read only one of the two definitions still exists, so no assert can compare
 * them. Change one, change the other. */
#ifndef SYS__HEAP_OBJECT__LOCK_TRIES
#define SYS__HEAP_OBJECT__LOCK_TRIES 64u
#endif
#ifndef SYS__HEAP_OBJECT__LOCK_BACKOFF_CYCLES
#define SYS__HEAP_OBJECT__LOCK_BACKOFF_CYCLES 10u
#endif

/* ── WHAT A DYING OBJECT HANDS OVER ─────────────────────────────────────────────────────────────────
 * A kind being taken apart says what it was holding by MOVING it onto the releaser stack it is handed.
 * Every row's method takes that one stack and answers nothing, so the ARC's rows all deal in the same
 * currency, and there is one shape in the whole unwinding rather than a container and a schedule.
 *
 * ⛳ A TRANSFER RATHER THAN A HAND-BACK, AND THE COUNT NEVER MOVES ON THE WAY: a transfer changes the
 * OWNER and not the number, so a value passes out of a dying container into the unwinding without an
 * atomic spent on the journey — where handing back a chain costs a hold taken and a hold given back for
 * every link, and gets the second wrong if the first failed.
 * ⛔ WHAT IT COSTS, SAID PLAINLY BECAUSE IT IS A REAL COST: a container being taken apart has its
 * contents COPIED onto that stack while its own room is still standing, so for that moment they exist
 * twice. A dying stack copies one reference, being one chunk however many values sit inside it; a dying
 * list chunk copies every cell it holds.
 *
 * The name below is the stack's chunk type, promised without its shape — the shape belongs to the file
 * that owns chunks.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
typedef struct sys__stack_chunk sys__stack_chunk;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_OBJECT_CUH */
