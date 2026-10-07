#ifndef SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../../manifest__header.cuh"   /* the list of packages, which the gathers below are driven by */
#include "../language_contract.cuh"
#include "heap_object__header.cuh"
#include "bindings__header.cuh"
#include "computing_base__header.cuh"
#include "fault__header.cuh"
#include "error.cuh"
#include "heap__header.cuh"
#include "list__header.cuh"
#include "opcodes/opcodes.cuh"
#include "node_array__header.cuh"
#include "system_register__header.cuh"
#include "silicon/silicon__header.cuh"
#include "stack__header.cuh"
#include "procedure.cuh"                 /* the contract row names its size */
#include "string.cuh"                    /* and so does this one */
#include "dictionary.cuh"                /* and these three */
#include "opcodes/opcodes_abi__header.cuh"  /* the verb adapters the dispatch switch names */
/* ══ the object interface, implemented ═══════════════════════════════════════════════════════════════
 *
 * Every `sys__heap_object__` verb that DOES something is defined here, together with the private ones its
 * surface is built from. That is the whole rule for this file, and it is a rule about the INTERFACE
 * rather than about lifetime — `clone`, `set` and the lock are here too, and none of them is a thing that
 * happens when a count runs out.
 *
 * ── WHY THE INTERFACE AND ITS IMPLEMENTATION ARE TWO FILES AND NOT ONE ──────────────────────────────
 * The declarations must come FIRST, because every file names them. The definitions must come LAST,
 * because two of them are switches over every kind — and a switch that calls a stack's method cannot be
 * written before the stack. One file cannot be both ends of an ordered include list, so it is two, and
 * the split is at the only place it can be.
 *
 * ⛔ WHAT IS IN THE HEADER INSTEAD, AND IT IS NOT AN EXCEPTION: the head LAYOUT — the count and the
 * lock word's index — is in the header beside the declarations, and so are the four reads that ask what
 * an object is, because a head is all any of them needs and containers writing a fresh one need them
 * long before any of this exists. Shape in the header, behaviour here.
 *
 * Letting go of a reference is the same three lines wherever it happens, and the interesting part is what
 * follows the last one. An object whose final reference goes has to be taken apart, and only something
 * that knows its shape can do that: a stack has a chain of chunks behind it, a list will have its own
 * arrangement, and the code letting go of the reference knows neither. So the object says what it is and
 * this looks the answer up.
 *
 * ── WHY THIS FILE COMES AFTER THE CONTAINERS ────────────────────────────────────────────────────────
 * It names them. A registry has to be written after the things it registers, which is the same reason the
 * package manifest is ordered rather than sorted, and it is why this is a file of its own rather than a
 * few lines added to the allocator: the allocator is what containers are built ON, and this is built on
 * containers.
 *
 * ── THE COUNTING ITSELF IS NOT HERE ─────────────────────────────────────────────────────────────────
 * It is in the heap, with the word it moves: an object carries its count in the node its allocation
 * begins with, beside its kind, so a holder that has the object has the count one node back. Taking a
 * hold and giving one up are both written there. What is left here is only what happens when a giving up
 * leaves none — which is a question about the object's KIND, and the heap is written before any kind
 * exists.
 *
 * ── AND THE LOCK IS ALREADY PAID FOR ────────────────────────────────────────────────────────────────
 * The atomic decrement that takes a count to zero establishes exactly one caller, so whatever runs next
 * runs alone without anything further being arranged. Taking an object apart is therefore free of
 * synchronisation rather than merely cheap — the hard part was bought by the decrement.
 *
 * ⛔ TWO FAILURES SHARE A SHAPE HERE AND ONLY ONE OF THEM IS SILENT, WHICH IS WORTH SEPARATING BECAUSE
 * THE LOUD ONE IS EASY TO MISTAKE FOR THE OTHER.
 *   a kind with NO ROW      RAISES. All three dispatches below end in a `default:` that reports
 *                           `UNKNOWN_KIND` rather than returning quietly, precisely so this cannot be
 *                           the silent case. It is a run-time refusal and not a compile error — that is
 *                           the one direction the mechanism cannot close — but it is a refusal.
 *   a row whose method is   LEAKS EVERYTHING THE OBJECT HELD, SILENTLY, WHILE EVERY COUNT BALANCES. The
 *   EMPTY                   dispatch found an arm, called it, and got back nothing to unwind. Only the
 *                           residue at the end of a launch shows it.
 * ⇒ ★ THE DANGEROUS ONE IS THE ROW THAT EXISTS AND DOES NOTHING, not the row that is missing. A row and
 * its object belong in one commit, and a `release` that legitimately hands over nothing should say so —
 * which is why `release__default` carries a sentence explaining that it is the whole truth for a
 * value-shaped object rather than a stub.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* Whether an offset can be one of ours at all.
 *
 * ⛔ THIS IS THE ONE PLACE THE WHOLE ARRANGEMENT COULD GO WRONG QUIETLY. Finding an object's count means
 * stepping one node back from its reference, which answers something for ANY offset — so an offset that
 * never came from a placement returns a node that is somebody else's data, and the count read out of it is
 * whatever those bytes happen to be. Refusing here turns that into a fault at the moment it happens
 * rather than a wrong count somewhere later. */
static __device__ inline bool sys__heap_object__zzpackage_addressable(uint64_t offset) {
    if (!sys__heap__zzpackage_published() || offset >= sys__heap__zzpackage_total_nodes()) return false;
    /* ⛳ TWO THINGS CAN BE REFERRED TO AND THEY SIT IN DIFFERENT PLACES. Anything CARVED begins past its
     * chunk's own header, because the first nodes of a chunk describe the chunk. The exception is the
     * chunk ITSELF, which is a registered kind whose head is the chunk's first node — so its reference is
     * the second, and that is the one offset inside a header that names something real.
     *
     * ⚠ AND IT IS A WEAK CHECK, WHICH IS THE PRICE OF LETTING A KIND CHOOSE ITS SIZE. If every allocation
     * were the same width a reference would be recognisable by its remainder, and any value without that
     * remainder would fail here. With widths differing there is no such arithmetic: what is left is a
     * bounds test and a header test, and a wrong offset landing inside a live object passes both. */
    const uint64_t within = offset & ((uint64_t)SYS__HEAP__CHUNK_NODES - 1ull);
    return within > (uint64_t)SYS__CHUNK__FIRST || within == 1ull;
}

static __device__ inline sys__heap_node sys__heap_object__reference_to(uint64_t offset) {
    sys__heap_node reference;
    reference.dtype    = SYS__KIND__OBJECT_REFERENCE;
    reference.num_args = 0u;
    reference.args[0]  = offset;
    return reference;
}

static __device__ inline sys__heap_node sys__heap_object__create_refusal(sys__heap_node* base,
                                                                    const sys__heap_node* parameters) {
    (void)base; (void)parameters;
    sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_CONSTRUCTIBLE);
    return sys__heap_node__nothing();
}

static __device__ inline sys__heap_node sys__heap_object__create__plain(sys__heap_node* base,
                                                                   const sys__heap_node* parameters) {
    (void)base; (void)parameters;
    sys__heap_node* head = sys__heap__make(SYS__KIND__HEAP_OBJECT);
    if (head == 0) return sys__heap_node__nothing();          /* the making raised the reason */
    return sys__heap_object__reference_to(sys__heap__offset(head + 1));
}

/* ── MAKING ONE FROM THE AST ─────────────────────────────────────────────────────────────────────────
 * The dispatch a program reaches. It asks the kind's row who builds one and hands the parameters
 * straight over; every arm is a wrapper around a constructor that exists in C on its own account, so
 * nothing here knows how any kind is put together.
 *
 * ⛔ AND THE KINDS THAT DO NOT SURFACE ARE IN THE LIST TOO, NAMING THE REFUSAL. That is deliberate and it
 * is not the same as leaving them out: an absent row falls to the `default` below, which cannot tell a
 * kind that MUST NOT be built from one somebody forgot to register. A named refusal says it was decided.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
static __device__ inline sys__heap_node sys__heap_object__create(sys__heap_node* base, sys__kind kind,
                                                            const sys__heap_node* parameters) {
    switch (kind) {
/* ⛳ IT NAMES UP TO THE COLUMN IT USES AND ABSORBS THE REST: a node count stands past the constructor
         * in every row, and the one consumer that reads it is the allocator's. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_ARM(PKG, DTYPE, RELEASE, PROVIDER, CLONE, CONSTRUCT, ...) \
        case DTYPE: return CONSTRUCT(base, parameters);
#define SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__OBJECTS(SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_ARM, NAME)
#define SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER_ROW(ID, NAME, ...)  SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER(NAME)
        PACKAGE_LIST(SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER_ROW)
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_ARM
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CREATE_GATHER_ROW
        default:
            /* Nothing said what this is, so nobody can be asked to build one. */
            sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_UNKNOWN_KIND);
            return sys__heap_node__nothing();
    }
}

/* Building one from a program: the FIRST parameter is the kind and the rest are the kind's own, which is
 * `create(car(list), cdr(list))` with the list still to come.
 *
 * ⛳ THAT CALL IS NOT IN THIS FILE: the published `create` is in `opcodes/opcodes_abi.cuh`, and it
 * hands `sys__heap_object__create` above a node it has filled.
 * ⛔⛔ AND THE `cdr` THERE IS A SHIFTED COPY, WHICH IS THE HONEST SHAPE OF THE LIMIT AND NOT A TRICK. A
 * node carries six arguments; taking the kind out of the first leaves five, so a kind wanting more than
 * five cannot be built this way and nothing pretends otherwise. ⛳ THE WAY OFF THE LIMIT IS A REAL
 * `cdr`: `sys__sublist__cdr` exists. What changes is what a constructor is HANDED — a list instead of a
 * node of five cells — which is that function and every row's constructor, and no prerequisite
 * underneath either. */
/* ── APPLYING A PUBLISHED VERB ───────────────────────────────────────────────────────────────────────
 * Where a call from the language arrives. It is the same gather as the kinds, over the other list, and it
 * is what makes the published rows real rather than a table nobody reads.
 *
 * ⛳ THE VERB IS PACKED, so a second package's words land here beside this one's without either knowing.
 * ⛔ AND AN UNKNOWN VERB RAISES rather than answering nothing, for the reason every dispatch in this file
 * raises: a quiet answer is indistinguishable from a verb that ran and returned nothing.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
static __device__ inline void sys__heap_object__apply(uint64_t verb, sys__heap_node* base,
                                                                    uint64_t form) {
    switch (verb) {
/* ⛳ IT NAMES THE NUMBER AND THE WRAPPER AND ABSORBS THE REST. ⚖ ARCHITECT, on this contract: *"i think
         * this contract wont change so it can stay without the ..."* A row carries one thing past the
         * wrapper — the opcode constant's name, which `SYS__LANGUAGE_CONTRACT__VERB_ID` publishes and a
         * dispatch has no use for — so the tail is what keeps a column this switch never reads from
         * reaching it.
         * ⛔ AND NOTHING ENFORCES THAT, ON EITHER LIST: gate rule ⑦ keeps only arms whose first parameter
         * is `DTYPE`, and every arm in this tree begins with `PKG`. `MEASURED`: a run of
         * `scripts/src_c_subset_gate.py` prints `ARMS 0 row consumer(s)`, so it sees neither this arm nor
         * the kind arms. The rule is in this comment or it is nowhere. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_VERB_ARM(PKG, BASE, NAME, APPLY, ...) \
        case PACKAGE_VERB(PKG, BASE): APPLY(base, form); return;
#define SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__VERBS(SYS__HEAP_OBJECT__ZZPACKAGE_VERB_ARM, NAME)
#define SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER_ROW(ID, NAME, ...)  SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER(NAME)
        PACKAGE_LIST(SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER_ROW)
#undef SYS__HEAP_OBJECT__ZZPACKAGE_VERB_ARM
#undef SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER
#undef SYS__HEAP_OBJECT__ZZPACKAGE_VERB_GATHER_ROW
        default:
            /* ⛔ AND IT LEAVES THE FORM UNTOUCHED, WHICH IS WHAT THE EVALUATOR NEEDS IT TO DO. A verb that
             * does not exist has not rewritten anything, so the scan sees the same form it handed over —
             * and that is a hang, not a wrong answer. The raise is what a reader has; making the form an
             * ERROR is what would let the program itself carry on, and it is owed. */
            sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_UNKNOWN_VERB);
            return;
    }
}

/* ── TAKING A HOLD ───────────────────────────────────────────────────────────────────────────────────
 * Become one of the holders of whatever lives at `offset`. Answers the count afterwards, so a caller can
 * tell a first holder from a second.
 *
 * ⭐ IT IS NOT "get the object" — the caller usually has it already. What it takes is a HOLD, and the
 * only thing that changes is how many there are, which is why the answer is a number and not a value.
 *
 * It sits beside the decrement below: the two move the same word in opposite directions.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint32_t sys__heap_object__retain(uint64_t offset) {
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_A_REFERENCE);
        return 0u;
    }
    sys__heap_node* head = sys__heap_node__zzpackage_head(offset);
    return sys__heap__zzpackage_take_up((unsigned int*)&SYS__HEAP_OBJECT__COUNT(head));
}

/* ── AND LETTING ONE GO, WHICH IS ALL THIS TIER KNOWS HOW TO DO ─────────────────────────────────────
 * The decrement, and the two ways it can be wrong. What comes after a count reaching zero is a question
 * about the object's KIND, and this file is written before any kind exists — so the answer is not here
 * and cannot be. `sys__heap_object__release` pairs this with the dispatch, and it is the only thing that
 * does.
 *
 * ⛳ THAT SPLIT IS WHY THIS IS NOT PUBLIC. Calling it alone takes an object's last hold away and then
 * walks off: the count says nobody wants it, its room is never given back, and everything it was holding
 * is unreachable while every counter reads healthy. There is one correct next line and the caller does
 * not have it.
 *
 * Answers false when it refused — the offset is not one this heap allocation pool handed out, or the count was already
 * at nothing — and raises which of the two it was.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline bool sys__heap_object__zzprivate_reference_decrement(uint64_t offset, uint32_t* left) {
    if (left == 0) return false;
    *left = 0u;
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_A_REFERENCE);
        return false;
    }
    sys__heap_node* head = sys__heap_node__zzpackage_head(offset);
    if (!sys__heap__zzpackage_give_up((unsigned int*)&SYS__HEAP_OBJECT__COUNT(head), left)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_OVER_RELEASED);
        return false;
    }
    /* A count reaching zero is the one event in the unwinding that nothing else records. Everything
     * after it — what the object held, where its room went — is decided by kind, and by then the number
     * that got us here is gone. */
    if (*left == 0u) sys__heap__zzpackage_record_deallocation();
    return true;
}

/* ── TAKING AN OBJECT FOR THE LENGTH OF A WRITE ──────────────────────────────────────────────────────
 * A counted object carries its own lock, in the head node, two words along from the count: the count
 * shares its word with the kind, the node count is the word between, and the lock is the head's first
 * argument. Everything that MODIFIES an object takes it first — and so does a read that takes more than
 * one step. `sys__stack__peek` is the case: it reads a cursor, then the cell under it, then takes a hold,
 * and a pop arriving between two of those empties the chunk and hands its room back underneath it. A read
 * that hands out no hold takes no lock, and owes the discipline its own comment states instead.
 *
 * ⚖ ARCHITECT: *"we need an atomiccas when doing a push pop replace or deallocate ... we could dedicate
 * an argument to it directly, to be in specific of the data type machinery."* And, on the surface:
 * *"we could also generalize this setup with sys__heap_object__lock(reference) and
 * sys__heap_object__lock_release(reference)."*
 *
 * ⭐ IT TAKES A REFERENCE, LIKE EVERY OTHER VERB IN THIS FAMILY. A caller holds a value, not a head — the
 * head is where the word lives, and finding it is one step back this can do for itself. That also means
 * the range check comes for free: a reference that could not be one of ours is refused here rather than
 * stepped into somebody else's memory and swapped. So it is not the stack's
 * lock or the list's — it is the object header's, at the same place in every kind, which is what lets
 * something holding an offset take it without knowing what is at the other end.
 *
 * ⛔ WHY IT IS A WHOLE WORD AND NOT THE BYTE THAT WOULD DO. There is no portable compare-and-swap
 * narrower than 32 bits — this package's seam offers `sys__silicon__cas_u32` and nothing smaller, because
 * the widths available genuinely differ per target. A byte would have to be swapped as part of the word
 * around it, and the only word next door holds the reference count, which retain and release update
 * atomically. Two independent atomics racing on one word is a livelock before it is a bug. The head has
 * forty spare bytes, so a word of its own costs nothing.
 *
 * ⛔⛔ AND THE ONE RULE THAT MAKES THIS SAFE, WHICH IS NOT ENFORCED HERE AND CANNOT BE:
 *
 *     YOU MAY ONLY LOCK WHAT YOU HOLD A REFERENCE TO.
 *
 * The lock lives INSIDE the thing it protects, so an object taken apart while somebody waits on its lock
 * leaves that waiter spinning on memory the heap allocation pool has already handed to someone else. Holding a reference
 * is what makes that impossible: two holders means the count cannot reach zero, so neither one's release
 * can free it out from under the other. Legitimate callers already satisfy this — they got the pointer
 * from somewhere, and getting it was taking a reference.
 *
 * ⇒ ⛳ AND IT IS WHY THE COLLECTOR TAKES NO LOCK. It runs when a count has reached zero, so by that rule
 * nobody can be holding one, and taking it would be machinery with no contender.
 *
 * ⛳ AND IT IS TAKEN ON AN IDENTITY, NEVER ON A PART OF ONE. ⚖ ARCHITECT: *"the stack chunks are internal
 * by definition, so the access control is moved at the holder."* A thing reachable only through another
 * thing does not need its own: holding the one that owns it excludes everybody from all of it, and a
 * second lock underneath would only ever be taken by whoever already held the first. **So no two of these
 * are ever held at once, and there is no order to get wrong.**
 *
 * ⛔⛔ WHAT IT EXCLUDES IS RUNNER THREADS. A block is one runner thread, so the contenders are other
 * blocks; nothing in the runtime detects a holder that never lets go — which is why the wait below is
 * BOUNDED and raises rather than spinning.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* The lock word itself: the low half of the head's first argument, because the swap is 32 bits wide and
 * the argument is 64. The other half is not used and is not read; the initialiser clears the whole word so
 * a fresh allocation does not begin holding whatever its last tenant left. */
static __device__ inline unsigned int* sys__heap_object__zzprivate_lock_word(sys__heap_node* head) {
    return (unsigned int*)&head->args[SYS__HEAP_OBJECT__LOCK];
}

/* Ask for the object, and do not come back until you have it or it is hopeless.
 *
 * ⚖ ARCHITECT, on what to do when the swap fails: *"make it wait 10 clock cycles plus its block id, so
 * waits happen at different speed and we resolve a potential deadlock."* That is the whole reason for the
 * block index in the delay — not fairness and not throughput. Two blocks that back off by the same amount
 * retry together forever; two that back off by different amounts cannot.
 *
 * ⛳ AND WHY THERE IS A TRY COUNT. An unbounded spin is a runner that hangs with no suspect; a bound
 * turns that into a fault with a name and a program counter. The number is a `REASONED` guess at
 * "longer than any legitimate holder", not a measurement — it has not yet been under contention between
 * runners. */
static __device__ inline bool sys__heap_object__lock(uint64_t offset) {
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_LOCK);
        return false;
    }
    unsigned int* word = sys__heap_object__zzprivate_lock_word(sys__heap_node__zzpackage_head(offset));
    for (uint32_t t = 0u; t < SYS__HEAP_OBJECT__LOCK_TRIES; ++t) {
        if (sys__silicon__cas_u32(word, SYS__HEAP_OBJECT__LOCK_FREE, SYS__HEAP_OBJECT__LOCK_HELD) == SYS__HEAP_OBJECT__LOCK_FREE)
            return true;
        sys__silicon__wait_cycles(SYS__HEAP_OBJECT__LOCK_BACKOFF_CYCLES + sys__silicon__block_id());
    }
    sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_LOCK);
    return false;
}

/* Hand it back. It is a swap and not a store so that releasing something not held is caught: a store
 * would make the mistake invisible and the next holder would find the object already free. */
static __device__ inline void sys__heap_object__lock_release(uint64_t offset) {
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_LOCK);
        return;
    }
    if (sys__silicon__cas_u32(sys__heap_object__zzprivate_lock_word(sys__heap_node__zzpackage_head(offset)),
                            SYS__HEAP_OBJECT__LOCK_HELD, SYS__HEAP_OBJECT__LOCK_FREE) != SYS__HEAP_OBJECT__LOCK_HELD)
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_LOCK);
}

/* ⛔ EVERY CLONE COLUMN BUT SUBLIST'S NAMES THIS ONE, SO THE DISPATCH REFUSES EVERY OTHER KIND — AND
 * THAT IS NOT BECAUSE NOTHING ELSE CAN BE COPIED. SUBLIST's row names `sys__list__zzpackage_clone_object`;
 * the other rows have no copier WIRED INTO THEM. So a refusal is a wiring state and not a capability
 * one, and the two read identically from here, which is why it is worth saying: a reader who takes this
 * for "the copiers do not exist" goes and writes one that already does.
 * ⛳ THE INTERFACE EXISTS SO THAT CHANGING THAT IS ONE ROW EDIT AND NO NEW MACHINERY, and so that a
 * caller asking for a copy of a kind that refuses is told no loudly rather than handed a zero it has to
 * interpret. */
static __device__ inline uint64_t sys__heap_object__clone__default(sys__heap_node* head,
                                                                          sys__heap_node* computing_base) {
    (void)head; (void)computing_base;
    sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_CLONABLE);
    return 0ull;
}

/* ── WHAT EACH KIND OF OBJECT DOES WHEN ITS LAST REFERENCE GOES ──────────────────────────────────────
 * A plain object holds nothing, so all it owes is its own room. That is not an empty method standing in
 * for one that should do something — it is the whole of what a value-shaped object owes.
 *
 * ⛔ AND GIVING THE ROOM BACK IS NOT THE KIND'S LINE. It is one line at the end of the dispatch that ran
 * the method, taken once for every kind, which is why no method here mentions a chunk: a method says what
 * its object was HOLDING and stops, and where that object sat is the allocator's business.
 * ⇒ ★ SO A NEW KIND'S `release` MUST NOT HAND ITS OWN CHUNK BACK. Doing it in the method as well as
 * centrally releases the same room twice, and the second hand-back is a chunk given to whoever asks next
 * while somebody is still carving in it.
 *
 * ⭐ AND THE ONE KIND THAT KEEPS ITS ROOM SAYS SO IN THE COUNT RATHER THAN IN A LIST OF WHO IS SPECIAL.
 * The central give-back is guarded by "is it still dead" — a count of zero — so a method that writes one
 * back has declined it and nothing central has to know which kind did.
 *
 * The computing base is that kind, and it is not a case but a mistake. It outlives everything it
 * names, so a release reaching it is an error, and an error that then handed the block's own chunk to the
 * heap allocation pool would be far worse than the one that caused it. Its method raises, hands nothing
 * over and writes the count back; it is in `computing_base__impl.cuh`, with the rest of what a computing
 * base does.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__heap_object__release__default(sys__heap_node* head,
                                                                uint64_t* releaser_stack) {
    (void)head; (void)releaser_stack;             /* it held nothing, so there is nothing to move */
}

/* ── THE DISPATCHES, GENERATED FROM THE MACROS ─────────────────────────────────────────────────────────
 * Each package keeps its kinds in one macro, in `language_contract.cuh`, and the switches that expand that
 * macro are here. Nothing pushes the two apart and nothing pulls them together: a macro body is only text
 * until something expands it, so the rows can live wherever their package wants them, and a switch needs
 * the methods a row names to have been DECLARED — which is what a header is for — and not defined. That
 * is why this file asks for the stack's header and not for the stack.
 *
 * ⚖⚖ AND CAN A SECOND PACKAGE ADD A KIND OF ITS OWN? ⚖ ARCHITECT: *"can we expand the x macro if we do
 * it this way and `nn` wants a special ast data type with its own release and deallocate method or
 * are we out of luck?"*
 *
 * The mechanism is that a switch never names a package. It PASTES the name onto `__LANGUAGE_CONTRACT__OBJECTS`
 * and expands whatever that turns out to be, so arms appear because a package's row exists and not
 * because somebody edited the switch. Both switches below are written that way, and so is the create
 * above. The name they paste comes from `PACKAGE_LIST`, which drives every gather in this file: adding a
 * package is adding a row to that list, and nothing here is touched.
 *
 * ⛳ A CASE LABEL IS NOT THE NUMBER WRITTEN IN THE PACKAGE, WHICH IS THE HALF OF THE ARRANGEMENT VISIBLE
 * FROM HERE. A row carries a package's OWN number, unique only within that package, and the machinery adds
 * the package id to it so that uniqueness survives leaving the package — the top byte of a kind is whose
 * it is. So the labels these switches emit are tagged and the rows they came from are not, and neither the
 * row-writer nor this file writes the tag. That is the point rather than a convenience: the one
 * arrangement already tried and measured is applying it per row, and 33 opcodes escaped untagged that
 * way. A switch that pastes cannot miss one.
 * ⛔ IT ALSO MEANS A KIND IS NOT DENSE AND THESE SWITCHES MUST NOT ASSUME IT IS. Two packages' kinds sit
 * 16,777,216 apart, so nothing here may become a table indexed by a kind or a test on a range of them.
 * Both switches compare for equality, which is what keeps that true rather than an intention.
 * ▶ The numbering contract in full is in `packages/manifest.cuh`.
 *
 * ⛳ AND A THIRD SWITCH DOES THE SAME THING SOMEWHERE ELSE, WHICH IS WORTH KNOWING FROM HERE BECAUSE IT
 * READS A COLUMN THIS FILE NEVER TOUCHES. A row's third column says WHICH CHUNK a kind is carved from:
 * the one holding what LASTS, the one holding what CHURNS, or neither, for a kind that is placed by hand
 * and never made. The allocator reads it at the moment of CREATION, before there is an object at all —
 * which is why it lives there and not here: this file has no business knowing where anything is carved
 * from, and the allocator has no business knowing how a kind is taken apart.
 *
 * ⛳ AND EVERY GATHER OF THESE ROWS WORKS THE SAME WAY — FIVE OF THEM, WHICH IS WHAT MAKES A KIND ARRIVE
 * COMPLETE: a release arm, a clone arm and a constructor here, a provider and a node count in the
 * allocator. A kind that reached some of them and not the provider would be releasable, clonable, and
 * impossible to create — refused at the moment of creation, having been registered correctly.
 * ⛔ AND THE ALLOCATOR'S FALLTHROUGH SAYS "NOBODY REGISTERED THIS", which is a different thing from a row
 * DECLARING that its kind is never made — the computing base declares exactly that, being placed once by
 * hand. Two situations, two values, two words raised.
 * ⇒ ★ BECAUSE A DIAGNOSTIC NAMES THE CONDITION IT FOUND AND NEVER THE REASON IT HOLDS. One word for both
 * would say a kind is not carved from a provider — true, and pointing at a row that is complete and
 * correct, so the one person able to fix it is sent to the one place that is fine.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

 
/* Take one dead object apart: ask it what it was holding.
 *
 * ⛳ AND ONLY THAT — A KIND'S METHOD SAYS WHAT WAS HELD AND NEVER WHERE THE ROOM GOES. The room is given
 * back in ONE place, at the bottom of this verb, and the two kinds that must not give theirs back are not
 * named there either: a chunk IS its own room so the arithmetic lands on the head itself, and the
 * computing base writes its count back rather than dying. Both answers come out of the object — the
 * question asked is "is it still dead" — so nothing here reads a list of who is special, and no method
 * has to be trusted to decline quietly. ⇒ ★ ASKING THE OBJECT BEATS ASKING THE KIND, because a list of
 * exceptions is a second place the truth lives and the only way to check it is to read every method.
 *
 * ⭐ WHAT COMES BACK IS A CHAIN, AND NOTHING IN IT HAS BEEN RELEASED. A kind hands over what it was
 * holding without touching a count: the values sit in the cells they always sat in, and the hold that
 * was the dying object's is now the chain's. So the answer is not a list of things to go and find — it
 * is the holding itself, moved. */
static __device__ inline void sys__heap_object__zzprivate_deallocate(uint64_t offset,
                                                                   uint64_t* releaser_stack) {
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_A_REFERENCE);
        return;
    }
    sys__heap_node* head = sys__heap_node__zzpackage_head(offset);
    switch (sys__heap_object__type_from_head(head)) {
/* ⛳ IT NAMES THE COLUMNS IT USES AND ABSORBS THE REST. A row will grow a column one day, and a consumer
         * that spelled out every one of them would have to be edited for a column it does not read. The
         * trailing `...` is what stops that, and it costs nothing: the columns are ordered commonest-first,
         * so what any consumer ignores is always a tail. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_ARM(PKG, DTYPE, RELEASE, ...) \
        case DTYPE: RELEASE(head, releaser_stack); break;
/* ⭐ ONE PACKAGE'S ARMS, BY NAME. A package is named in lowercase where packages are listed, so its
         * macro is `<name>__LANGUAGE_CONTRACT__OBJECTS`, and pasting the name is what lets this switch be
         * written once and never edited as packages arrive. `MEASURED` to expand correctly, with arms from
         * two packages landing in one switch.
         * ⛳ THE NAME IS ALL IT TAKES, and that is worth one line of its own: the path to a package's
         * header is a column this switch never reads, and its id is one it never reads either — an
         * OBJECTS row names the PUBLISHED kind, which the KINDS gather already tagged, so the row
         * supplies one token that is the whole answer and there is nothing left here to get wrong.
         * ⛔ AND THE ARM MUST NOT TAG IT AGAIN, and the reason that would be invisible is the reason
         * it is worth a line: `sys` is package 0, so a second application adds zero. `nn`'s OBJECTS
         * rows would compute 0x01000000 + 0x01000000, match no arm, and take the `default` for
         * release, clone, provider and size — silently, for every object of that kind. ⚖ RULED: the
         * tag is applied once, where the constant is defined. ▶ the note at `PACKAGE_DTYPE` in
         * `packages/manifest__header.cuh`. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__OBJECTS(SYS__HEAP_OBJECT__ZZPACKAGE_ARM, NAME)
        /* ⛳ AND THE ADAPTER THE LIST OF PACKAGES DRIVES, which takes a ROW and reads the one column it
         * needs. This switch names no package: adding one is adding a row to that list, and nothing in
         * this file is touched. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_GATHER_ROW(ID, NAME, ...)  SYS__HEAP_OBJECT__ZZPACKAGE_GATHER(NAME)
        PACKAGE_LIST(SYS__HEAP_OBJECT__ZZPACKAGE_GATHER_ROW)
#undef SYS__HEAP_OBJECT__ZZPACKAGE_ARM
#undef SYS__HEAP_OBJECT__ZZPACKAGE_GATHER
#undef SYS__HEAP_OBJECT__ZZPACKAGE_GATHER_ROW
        default:
            /* Nothing said what this is, so nothing can be said about what it held. Raising is the only
             * honest answer: the alternative is to return quietly and leave a leak that every counter
             * reports as healthy. */
            sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_UNKNOWN_KIND);
            return;
    }

    /* ⭐⭐ AND THE ROOM GOES BACK HERE, ONCE, FOR EVERY KIND — WHICH IS WHY NO KIND'S METHOD MENTIONS A
     * CHUNK. ⚖ ARCHITECT: *"it is simply managed in terms of create and release of the object
     * and the heap is abstracted away for them."* Somebody writing a kind says what it was holding; where
     * it sat is the allocator's business and was never theirs to get right.
     *
     * ⭐ THE TEST IS "IS IT STILL DEAD", WHICH IS ONE QUESTION INSTEAD OF A LIST OF WHO IS SPECIAL. Two
     * kinds must not give their room back at this moment, and the alternative to asking is for each of
     * them to decline quietly inside its own method — where the only way to find out is to read every
     * method. Both are saying the same thing: I AM NOT DEAD. The computing base writes one back because a
     * release reaching it is a mistake and it outlives everything. So the rule reads off the object
     * rather than off a list of who is special.
     *
     * ⛳ AND THE OTHER GUARD IS ARITHMETIC RATHER THAN A KIND. A chunk does not sit in a chunk, it IS one,
     * so the mask lands on the head itself — and a chunk that reached here has already gone home through
     * its own row. Nothing has to ask what it is looking at. */
    if (SYS__HEAP_OBJECT__COUNT(head) == 0u) {
        sys__heap_node* chunk = sys__heap_chunk__zzpackage_from_object_head(head);
        if (chunk != head) (void)sys__heap_chunk__zzpackage_release(chunk);
    }
}

/* Make a copy of one, into the block's own heap: ask it how.
 *
 * ⛔ EVERY ROW BUT SUBLIST'S REFUSES, AND THAT IS THE POINT OF HAVING THE DISPATCH ANYWAY: it makes a
 * kind clonable a ONE ROW EDIT rather than a new mechanism, and a request for a copy of a kind that
 * refuses fails loudly instead of returning a zero a caller has to interpret.
 *
 * ⭐ IT TAKES THE COMPUTING BASE BECAUSE A CLONE IS LOCAL. 
 * The point is that a clone should live in the heap assigned to the block's computing base to minimize 
 * contention and make reads more clustered.
 * A sender can then let go the moment a receiver has taken a reference.
 *
 * ⇒ ★ IT BUYS THE SENDER ITS MEMORY BACK, AND THE REASON IS DRAIN-ONLY. A chunk goes home only when
 * everything in it has died. A copy carved from the SENDER's chunk would hold that chunk open for as long
 * as the receiver kept it — the sender's reference could go and its CHUNK could not, so the whole of what
 * the sender was carving stays pinned by one value somebody else is still reading. Carving from the
 * receiver is what stops one block's liveness from being decided by another's.
 *
 * ⛳ AND IT ANSWERS AN OFFSET, NOT A POINTER, for the reason every reference here is one: a value has to
 * survive being copied into a cell some other block will read. */
static __device__ inline uint64_t sys__heap_object__clone(uint64_t offset, sys__heap_node* computing_base) {
    if (!sys__heap_object__zzpackage_addressable(offset)) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NOT_A_REFERENCE);
        return 0ull;
    }
    sys__heap_node* head = sys__heap_node__zzpackage_head(offset);
    switch (sys__heap_object__type_from_head(head)) {
/* ⛳ AND THIS ONE ABSORBS TOO, WITH A CONSTRUCTOR AND A NODE COUNT STANDING PAST THE COLUMN IT READS.
         * ⚖ ARCHITECT: *"can you add ... so it does not collapse if we add a field to the macro?"* So a
         * column appended to a row costs nothing anywhere: all three consumers take what follows and
         * ignore it.
         * ⚠ A `...` HANDED NOTHING IS ONE DIAGNOSTIC, UNDER `-pedantic` AND NOWHERE ELSE, because it is
         * ill-formed before C++20 and this tree builds at C++17 — four places in the build say so, and the
         * toolchain's own default is C++17. `MEASURED` on both compilers: the identical line is clean at
         * C++20, and it is quiet wherever a column stands past the ones an arm names. This arm is quiet.
         * What pays is whichever arm reads the LAST column, which is the allocator's node count.
         * ⇒ ★ SO THE PRICE IS ALWAYS EXACTLY ONE ARM — THE ONE AT THE END OF THE ROW — AND APPENDING A
         * COLUMN MOVES IT RATHER THAN RETIRING IT, which is the right way round for a guard: it sits on
         * the single consumer a new column would otherwise break. */
#define SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_ARM(PKG, DTYPE, RELEASE, PROVIDER, CLONE, ...) \
        case DTYPE: return CLONE(head, computing_base);
#define SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__OBJECTS(SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_ARM, NAME)
#define SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER_ROW(ID, NAME, ...)  SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER(NAME)
        PACKAGE_LIST(SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER_ROW)
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_ARM
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER
#undef SYS__HEAP_OBJECT__ZZPACKAGE_CLONE_GATHER_ROW
        default:
            /* Nothing said what this is, so nothing can be said about how to copy it. */
            sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_UNKNOWN_KIND);
            return 0ull;
    }
}

/* ── GIVING SOMETHING UP, AND FINISHING THE JOB ──────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"i want something that i can call directly as sys__heap_object__release(reference) and this
 * method will have the internal loop, where we call each object's release that does return the stack of
 * objects that also need releasing."* And on the shape of the loop itself: *"every deallocation produces
 * a stack of objects to decrement and we can have a stack of deallocation stacks, we consume one and we
 * get the return stacks pushed to that stack of stacks, and when we finish to release the current stack
 * we can go and pick another stack from the stack of stacks."*
 *
 * ⭐ ONE PUBLIC VERB, AND EVERY DECREMENT IN THE WHOLE UNWINDING HAPPENS INSIDE IT. Every call to the
 * private decrement there is stands in the function below — in this file and in the tree — so nothing
 * outside it can let a hold go. A kind being taken apart hands over what it held and touches nothing;
 * this pops one value out of the chain it is working through, lets go of it, and if that was its last
 * hold takes delivery of whatever IT held. No kind has to know that an unwinding is in progress.
 *
 * ⭐⭐ AND A DYING CONTAINER MOVES ITS CONTENTS ONTO THIS ONE STACK RATHER THAN HANDING BACK A CHAIN OF
 * ITS OWN. ⚖ ARCHITECT: *"maybe what we need really is a transfer method in the stack, where it does not
 * change the arc count and allows us to pass the ownership from the whatever type object to the stack.
 * yea this is better than having n subcases and it is usable by objects too."*
 * ⇒ There is one stack, so there is nothing to join and nothing to park. A stack OF stacks needs a chain
 * taken delivery of, a chain parked as a value, and the parker's own hold brought back down — three
 * moving parts that a single stack does not have. The depth was never what made it hard; joining two
 * chains was, and a transfer joins nothing.
 * ⛳ AND THE COUNT NEVER MOVES ON THE WAY. A transfer changes the OWNER and not the number, so a value
 * passes from a dying container into this stack and out again into the decrement below without one
 * atomic being spent on the journey — where handing over a chain would cost a hold taken and a hold
 * given back for every link, and get the second one wrong if the first had failed.
 *
 * ⛔ WHAT IT COSTS, AND IT IS A REAL COST RATHER THAN A ROUNDING ONE. ⚖ ARCHITECT: *"it means that stacks
 * lists and arrays hold twice the space until they are deallocated. i think it is an honest tradeoff and
 * makes the program readable, if it becomes a problem we will investigate but they are less than one heap
 * chunk each so i doubt it."* A container being taken apart has its contents copied into this stack while
 * its own room is still standing, so for that moment they exist twice.
 *
 * ⛳ AND A VALUE IS ONLY LOOKED AT ONCE. Popping transfers the hold out of the chain, so nothing is left
 * naming what is about to be let go of, and the same object cannot be reached twice.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ⭐⭐ KEPT OUT OF LINE, AND THE REASON IS THE SHAPE OF ITS CALLERS RATHER THAN ITS OWN SIZE.
 * ⚠ AND IT IS NOT THE ONLY ONE. Every published verb's apply wrapper, the sized carve, the kinds'
 * release internals and the fault channel are out of line too, each for its own reason.
 * ⇒ ★ DO NOT WRITE A COUNT OF THEM HERE: it is a claim about every other file in the package, and
 * nothing re-derives it. This drains a worklist, so it IS a loop — and a hold is let go in
 * dozens of places, among them the evaluator's own scan, which is also a loop. Inlined, this loop
 * becomes a loop inside each of those, and the innermost condition is a node whose address the body
 * hands to calls that write through it. A compiler asked to reason about a nest like that does not
 * decline: it tries, and it can try for a very long time. Kept out of line, a caller sees a call and
 * the nest never exists.
 * ⛳ WHAT IT COSTS, SAID PLAINLY: a call on every release that reaches zero. What that buys or costs
 * on the host is `ASSUMED` rather than measured; the figures behind the choice are device figures
 * (▶ the note at the top).
 * ⚠ THE MARKER IS KEPT AS A GUARANTEE RATHER THAN AS A MECHANISM, so that a future version of this
 * function small enough to tempt the inliner does not silently start being inlined into every teardown
 * site.
 * ⛳ WHETHER OUT OF LINE IS THE RIGHT TRADE AT ALL IS AN OPEN QUESTION AND NOT A SETTLED ONE: a spill
 * is a cost and so is a call, and which is cheaper here cannot be read off a spill count. It needs a
 * program worth timing on the host. */
static __device__ __noinline__ uint32_t sys__heap_object__release(uint64_t offset) {
    if (!sys__heap__zzpackage_published()) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_NOT_PUBLISHED);
        return 0u;
    }
    uint32_t left = 0u;
    if (!sys__heap_object__zzprivate_reference_decrement(offset, &left)) return 0u;   /* it said which way it was wrong */
    if (left != 0u) return left;                                /* somebody else still wants it */

    /* Nothing yet, and on most releases nothing ever: a dying object that held nothing moves nothing, so
     * the slot stays empty and the whole teardown allocates not one node. The first thing moved is what
     * turns this into a stack. */
    uint64_t releaser = 0ull;
    sys__heap_object__zzprivate_deallocate(offset, &releaser);

    /* ⭐⭐ ONE LAP, AND EACH PASS DOES ONE OF TWO THINGS: it takes the next thing off the stack, or —
     * when there is nothing left to take — it ends the stack itself and picks up whatever that ending
     * moved. So a hand-over that produces another stack is another pass rather than another frame,
     * and nothing here calls this function again.
     * ⛔ THE DISTINCTION THAT MATTERS IS BETWEEN THE DATA AND THE CODE. A stack that has been drained
     * holds nothing, so on every teardown anyone has yet written the outer lap runs exactly once and
     * the second pass finds nothing to do — that is true, and it is a fact about the VALUES. Whether
     * this function can reach itself is a fact about the TEXT, and only the second one is visible to a
     * compiler: one call back into here makes the emitted graph cyclic, and a cyclic graph cannot be
     * inlined and cannot have its stack bounded.
     * ⛳ THE COST WAS LARGE ON THE DEVICE (▶ the note at the top) and is `ASSUMED`, not measured, on
     * the host: a cycle stops inlining and stack bounding on any compiler. */
    /* ⛳ AND IT IS ONE LOOP RATHER THAN A LOOP INSIDE A LOOP, WHICH IS A CHOICE ABOUT WHAT THE
     * COMPILER IS ASKED TO PROVE. This slot is the loop's own condition, and the disposal in the body is
     * handed its ADDRESS — so it is free to write through the very thing being tested. A nested pair
     * would make it the
     * condition of both, so reasoning about either one would mean reasoning about everything those
     * calls do to it, and what they do is the whole of the teardown dispatch. Flat, there is one
     * condition and one body. */
    while (releaser != 0ull) {
        if (!sys__stack__zzpackage_empty_at(releaser)) {
            /* ⭐⭐ THE POP THAT DISPOSES OF NOTHING, AND IT IS WHAT KEEPS THIS FUNCTION OUT OF ITS OWN
             * CALL GRAPH. Taking the last value out of a worklist chunk leaves that chunk empty, and
             * the ordinary pop would give it up — which runs this function. So the chunk is handed back
             * instead and ended right here, with the same two steps every other death in this loop
             * gets. ⇒ ★ THE COLLECTOR IS THE ONE CALLER THAT CANNOT ASK FOR A THING TO BE COLLECTED. */
            uint64_t spent = 0ull;
            const sys__heap_node value = sys__stack__zzpackage_pop_deferring_at(releaser, &spent);
            if (spent != 0ull) {
                uint32_t chunk_left = 0u;
                if (sys__heap_object__zzprivate_reference_decrement(spent, &chunk_left) && chunk_left == 0u)
                    sys__heap_object__zzprivate_deallocate(spent, &releaser);
            }
            if (!sys__heap_node__carries_reference(value.dtype)) continue;

            uint32_t rest = 0u;
            if (!sys__heap_object__zzprivate_reference_decrement(value.args[0], &rest)) continue;
            if (rest != 0u) continue;
            sys__heap_object__zzprivate_deallocate(value.args[0], &releaser);
            continue;
        }

        /* Nothing left in it, so the stack itself goes — and ending it is the same two steps as
         * ending anything the passes above found: let go, and hand over what it held if that was the
         * last hold. Whatever it moves becomes the next thing to drain.
         * ⛳ ITS OFFSET IS TAKEN FIRST, because the hand-over writes through the very slot being read:
         * the sink is cleared before the call so that what comes back is what THIS stack moved, and
         * reading the offset afterwards would be reading whatever replaced it. */
        const uint64_t drained = releaser;
        uint32_t       held    = 0u;
        releaser = 0ull;
        if (sys__heap_object__zzprivate_reference_decrement(drained, &held) && held == 0u)
            sys__heap_object__zzprivate_deallocate(drained, &releaser);
    }
    return 0u;
}

/* ── RELEASING THE OBJECT A CELL NAMES, AND CLEARING THE CELL ────────────────────────────────────────
 * ⛳ THE NAME SAYS WHERE IT READS FROM, NEVER WHAT IT LETS GO OF. ⚖ ARCHITECT: *"we still release the
 * object but from the reference instead of by reference, and we are not releasing just the cell as this
 * implies."* A reference is not a counted thing and cannot be released; what it NAMES is, and it is
 * cleared afterwards because a reference left behind cannot be told from a live one.
 *
 * ⛳ AND THE PACKAGE HAS THREE WORDS FOR THREE THINGS, WHICH IS WORTH HAVING IN ONE PLACE BECAUSE TWO OF
 * THEM WERE USED FOR EACH OTHER:
 *     an ADDRESS      a real pointer. `sys__heap__offset` takes one.
 *     an OFFSET       a `uint64_t` measured from the heap base address. What every verb here takes, and what
 *                     `sys__heap__object_full_address` turns back into an address.
 *     a REFERENCE     a NODE that names an object: its tag says so and its first argument holds the
 *                     offset. What a caller usually has, and what the stack's surface already called one.
 * ⛔ CALLING AN OFFSET A REFERENCE MAKES THE WORD MEAN BOTH THE NUMBER AND THE NODE THAT HOLDS IT, and
 * nothing fails — the type is the same either way, which is exactly why it goes unnoticed.
 * ⚖ ARCHITECT: *"it should be sys__heap_object__release(holder) because you never know who else is holding
 * it. when release reaches 0 holders it will deallocate."*
 *
 * ⭐ THERE ARE TWO OF THESE AND NEITHER IS THE SAFE VERSION OF THE OTHER — THEY ARE FOR TWO DIFFERENT
 * SITUATIONS. `sys__heap_object__release` takes the OFFSET, which is what a caller has when it holds one
 * loose; another package with an offset in hand has no cell to reach through, and shutting it out would
 * be wrong. This one takes the CELL the reference lives in.
 *
 * ⛳ THE GUIDANCE IS ONE LINE: if you have a cell, use this one. Releasing through the reference instead
 * leaves the cell naming what may now be gone, and the clearing becomes yours to remember.
 *
 * ⛳ AND IT CLEARS THE CELL, WHICH IS THE HALF THAT CANNOT BE LEFT OUT. Releasing through the reference
 * alone leaves the cell still naming what may now be gone — and a stale reference is indistinguishable
 * from a live one: same dtype, same offset, and the next read steps into memory the heap allocation pool
 * has handed to somebody else.
 *
 * ⛔ IT IS A RELEASE AND NOT A DESTROY, WHICH IS THE ARCHITECT'S POINT: you never know who else is holding
 * it. The count goes down by one; if that leaves none, the object's own method runs and everything it held
 * goes with it. If it leaves some, nothing happens except that this caller is finished.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline void sys__heap_object__release_from_reference(sys__heap_node* reference) {
    if (reference == 0) { sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE); return; }
    if (sys__heap_node__carries_reference(reference->dtype)) (void)sys__heap_object__release(reference->args[0]);
    reference->dtype = SYS__KIND__VALUE_NULL;
    reference->args[0] = 0ull;
}

/* ── WRITING A CELL, WHICH IS THE ONLY SAFE WAY TO OVERWRITE ONE ─────────────────────────────────────
 * A cell that holds a reference owes a release before anything else can be written over it, and the new
 * value owes a retain. Doing that by hand is two things to remember in the right order; doing it here is
 * the assignment a program actually means.
 *
 * ⭐ IT IS WHAT ENDS A STACK, AND WHY THERE IS NO `sys__stack__release`. ⚖ ARCHITECT: *"you should call
 * sys__heap_object__release(reference) to get rid of an ast node object."* Right — and the
 * word in that sentence is doing two jobs: `release` takes the OFFSET, while a caller usually holds a
 * REFERENCE, the node that carries one. Releasing through the offset leaves that node naming what is
 * gone, for the reason argued directly above, so the verb takes the node — and being finished with a
 * stack is writing nothing over it.
 *
 * ⛳ AND IT IS THE SAME PRIMITIVE `replace` USES. Swapping the top of a stack is exactly
 * this on the cell the cursor points at — one rule, written once, rather than a rule the containers each
 * remember.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline void sys__heap_object__set(sys__heap_node* cell, const sys__heap_node* value) {
    if (cell == 0 || value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return;
    }
    /* The new one is taken BEFORE the old one is let go, because they can be the same object: releasing
     * first would take it to zero and take it apart, and the write would then store a reference to
     * something already handed back. */
    if (sys__heap_node__carries_reference(value->dtype)) (void)sys__heap_object__retain(value->args[0]);
    if (sys__heap_node__carries_reference(cell->dtype))  (void)sys__heap_object__release(cell->args[0]);
    *cell = *value;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__IMPL_CUH */
