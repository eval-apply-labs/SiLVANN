#ifndef SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "fault__header.cuh"            /* how it says a caller named a row that does not exist */
#include "heap_object__header.cuh"       /* the word a call handed no form refuses with */
#include "node_array__header.cuh"      /* the storage, and the three verbs that reach into it */
#include "heap__header.cuh"             /* the empty node a refusal answers with */
#include "system_register__header.cuh"  /* the rows, the count, and the verbs it fills in */

/* ── THE ONE WORD ────────────────────────────────────────────────────────────────────────────────────
 * Where the register's array begins, and the whole of what makes the register different from any other
 * array in the program. It is private to this file: every tier reaches a row through the verbs below,
 * because a name that can be indexed can be indexed wrongly and there is one place worth checking that.
 *
 * ⭐ A STATIC, BECAUSE THE ALTERNATIVE IS A CIRCLE. A block that had to be HANDED the register could only
 * be handed it by something that had already found it, and the register is what finding starts from. One
 * word with an address the compiler knows breaks that, and it is the only thing here that has to be
 * special — the array it names is an ordinary object, counted and released like every other, except that
 * nothing ever releases this one.
 *
 * ⛳ AND IT IS SHARED BY THE WHOLE DEVICE, WHICH IS WHAT MAKES A ROW WORTH HAVING AND ALSO WHAT MAKES A
 * ROW EXPENSIVE. One store is visible to every block; that is the point of the object. It is also why
 * rows are written at boot and read afterwards rather than being a place to keep something that moves —
 * nothing here orders two writers against each other, and nothing here is meant to.
 * ⛳ THE DISCIPLINE IS ABOUT *WHEN*, NOT *WHERE* — **written at boot, read afterwards** — so it holds
 * the same for RUNNER THREADS in one address space as for blocks on a card.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ uint64_t sys__system_register__zzprivate_base;

static __device__ inline bool sys__system_register__published(void) {
    return sys__system_register__zzprivate_base != 0ull;
}

/* ⭐⭐ THE BOOT OWNS THE STORAGE AND HANDS IT IN — ⚖ RULED BY THE ARCHITECT, AND THE REASON IS
 * WHAT THIS REGISTER IS FOR: its first row is where every block's computing base lives, and carving
 * anything from the heap is what asks a block's computing base where to carve. A register that
 * allocated its own room would therefore need the answer in order to be built, and the answer is the
 * thing it is being built to hold.
 * ⇒ ★ A STORE CANNOT BE ALLOCATED THROUGH THE LOOKUP IT IS GOING TO ABSORB. The two other ways out both
 *   keep the circle and hide it — a special case for the first block, or a boot that runs in two halves
 *   with an ordering nobody can see. Being handed the room removes it instead: there is no allocation on
 *   this path at all, so there is nothing for a lookup to happen inside.
 * ⛳ AND IT IS WHAT `package_init` IS FOR, which is the other reason to prefer it: storage a package owns
 *   and the engine does not, standing up before the first line of a program runs.
 *
 * ⛔ A SECOND CALL IS REFUSED RATHER THAN SERVED, and the reason is not tidiness: it would leave the
 * first array holding rows nothing can reach any more, alive forever because nothing releases the
 * register. One quiet leak of everything the register was naming, at boot, on a path nobody runs twice
 * on purpose — which is exactly the kind that survives to production. */
static __device__ inline bool sys__system_register__publish(sys__heap_node* room,
                                                            sys__heap_node* bases_room,
                                                            uint32_t blocks) {
    if (sys__system_register__published()) return true;
    /* ⭐⭐ THE FIRST ROW'S ARRAY IS STOOD UP HERE RATHER THAN BY THE CALLER, AND THAT IS THE COHESIVE
     * PLACE FOR IT. The boot supplies room and knows how many blocks there will be; it does not know
     * what a row means or what shape the thing a row names has. Placing an array is also package
     * business — it is how an object gets a head — so a caller that could do it would be a caller
     * reaching inside a kind. ⇒ ★ THE BOOT OWNS THE MEMORY; THE PACKAGE OWNS WHAT IS IN IT.
     * ⛳ AND THE ORDER INSIDE IS FORCED: the array has to exist before a row can name it, and the row
     * cannot be written before the register does. Two placements and one write, in the only sequence
     * that works, in the one function that knows all three. */
    const uint64_t bases = sys__node_array__zzpackage_place(bases_room, (uint64_t)blocks);
    if (bases == 0ull) return false;           /* the placing raised, and said which of its reasons */
    const uint64_t at = sys__node_array__zzpackage_place(room, (uint64_t)SYS__SYSTEM_REGISTER__ROWS);
    if (at == 0ull) return false;
    sys__system_register__zzprivate_base = at;
    /* ⛳ AS A PLAIN NUMBER AND NOT AS A COUNTED REFERENCE, WHICH IS A DECISION AND NOT AN OVERSIGHT. A
     * row holding something counted is retained by every read and released by every consumer — right
     * for a row a program writes, wrong for one the allocator reads on the way to every allocation,
     * where it would put two count updates on the hottest path there is. What is placed in the boot's
     * room is perennial by construction and has nothing to count. */
    if (!sys__system_register__set(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY,
                                   SYS__KIND__VALUE_INT, bases)) {
        sys__system_register__zzprivate_base = 0ull;   /* leave nothing half-standing */
        return false;
    }
    return true;
}

/* ── AND STANDING IT DOWN AGAIN ──────────────────────────────────────────────────────────────────────
 * Forget where the register is, so that the next `publish` stands a new one up instead of answering that
 * there already is one.
 *
 * ⛔⛔ IT EXISTS BECAUSE A BASE IS AN OFFSET AND AN OFFSET OUTLIVES NOTHING. The register lives in room the
 * boot set aside past the pool and inside the same allocation — where the allocator can never hand it out
 * and the offset is still a real subtraction from the base the allocator was given — so the moment that
 * allocation goes back the base names memory that belongs to somebody else, and it is still a perfectly
 * plausible number. `publish` refusing a second call is right for a machine that is UP, where standing a
 * second register up would orphan the first; it is exactly wrong across a shutdown, where there is no
 * first one left to orphan. Nothing inside this file can tell those two apart, so the boot says which by
 * calling this.
 * ⇒ ★ A "THIS IS ALREADY DONE" ANSWER IS ONLY TRUE WHILE THE THING IT DID STILL EXISTS.
 *
 * ⛳ NOTHING IS RELEASED AND NOTHING IS WALKED. What the rows named was carved from the same pool that is
 * going, so there is nothing to give back to an allocator that is itself being stood down. This is the
 * one teardown in the package that is correct precisely because it does nothing. */
static __device__ inline void sys__system_register__zzengine_retire(void) {
    sys__system_register__zzprivate_base = 0ull;
}

/* One place asks the two questions, so the verbs below cannot disagree about what a row is or about
 * whether there is a register at all. It raises where it refuses rather than leaving that to each caller,
 * because a refusal reported in three places is a refusal that gets reported in two of them.
 *
 * ⛳ AND IT ASKS THE BOUND ITSELF RATHER THAN LETTING THE ARRAY ASK IT. The array would refuse the same
 * index and raise its own word — correctly, and about the wrong thing: a caller naming a row that does
 * not exist has made a REGISTER mistake, and a log that says "an index outside an array" sends whoever
 * reads it looking at the storage instead of at the name. Same refusal, and the word says whose. */
static __device__ inline bool sys__system_register__zzprivate_reaches(uint32_t row) {
    if (!sys__system_register__published()) {
        sys__fault__raise(0ull, SYS__SYSTEM_REGISTER__FAULT_NOT_PUBLISHED);
        return false;
    }
    if (row >= (uint32_t)SYS__SYSTEM_REGISTER__ROWS) {
        sys__fault__raise(0ull, SYS__SYSTEM_REGISTER__FAULT_NO_ROW);
        return false;
    }
    return true;
}

static __device__ inline sys__heap_node sys__system_register__get(uint32_t row) {
    if (!sys__system_register__zzprivate_reaches(row)) return sys__heap_node__nothing();
    /* A row naming an object comes back HELD, because the array's get is where that is decided and a
     * register row is an element like any other. The register being perennial says nothing about what a
     * row holds — a caller reading one and using it a moment later must not be racing the next writer. */
    return sys__node_array__get(sys__system_register__zzprivate_base, (uint64_t)row);
}

static __device__ inline sys__kind sys__system_register__type(uint32_t row) {
    /* ⛳ THE ANSWER ON A BAD ROW IS A THIRD ONE ON PURPOSE. Answering the empty type would tell a caller
     * that the row exists and holds nothing, which is a sentence about a row that is not there — and it
     * is the reading that turns a mistyped index into a quiet zero somewhere much later. */
    if (!sys__system_register__zzprivate_reaches(row)) return SYS__KIND__INVALID;
    return sys__node_array__type(sys__system_register__zzprivate_base, (uint64_t)row);
}

static __device__ inline bool sys__system_register__set(uint32_t row, sys__kind dtype,
                                                        uint64_t value) {
    if (!sys__system_register__zzprivate_reaches(row)) return false;
    /* ⭐ THE COUNTING IS THE ARRAY'S AND NOT THIS FILE'S, which is what keeps a row's rules the same as
     * every other element's: the new one is taken before the old one is let go, because they can be the
     * same object and releasing first would take it apart and then store a reference to the room.
     * ⚠ AND NOTHING HERE IS ATOMIC, WHICH THIS DOES NOT PRETEND OTHERWISE ABOUT. A block reading a row
     * while another writes it is given no promise at all. `REASONED`, on the premise that rows are
     * written before the blocks that read them start. What would falsify it: a row written while the
     * grid is running, which needs a fence this does not have and is a different design rather than a
     * stronger version of this one. */
    sys__heap_node incoming = sys__heap_node__nothing();
    incoming.dtype   = dtype;
    incoming.args[0] = value;
    return sys__node_array__set(sys__system_register__zzprivate_base, (uint64_t)row, &incoming);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__IMPL_CUH */
