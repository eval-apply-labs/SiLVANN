#ifndef SILVANN__PACKAGES_SYS_CPU_STACK__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_STACK__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh"
#include "heap__header.cuh"
#include "stack__header.cuh"
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Nothing in it is needed to read, use, or maintain the code below; every entry is either a comparison against code that will not
 * exist by then, or a restatement of something the prose below already says. It is here only so an
 * agent working during the port does not re-derive a settled decision or re-propose a refused one.
 *
 * THE UNLOCKED PUSH ON A HOST. Its reachability argument costs the move to runner threads nothing, but
 *   there will be more callers that could break it, so the falsifier is easier to trip, not different.
 *   The lock figure that stood beside it — `MEASURED` 2,255 cycles a take-and-release, twice per drained
 *   object, 13.1% of everything a teardown spent — is a `k_eval` figure and will not reproduce on a CPU.
 *   It is kept here and nowhere else.
 * THE FAULT CODES. `transfer` checked the heap's launch conditions on every push until those were split
 *   into HPNP and HPNB; `sys__heap__make` names the cause since, so the per-push check went.
 * PEEK'S LOCK. Readers did not lock at first; the lock was added for the three-step race described
 *   over `sys__stack__peek`.
 *
 * RETIREMENT: delete when the port is complete and the old tree is gone. No triage required — if an
 * entry below ever becomes load-bearing for a reader, move it into the prose instead of keeping it
 * here, because this block is deleted wholesale and without being read.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * WHY A NEW TYPE RATHER THAN THE EXISTING PAGED LIST.
 *   Architect: "the paged list earns its keep when it can have adds in the middle; since this one is
 *   strictly accessed from the edge a simple array is cheaper." The list's normalize loop was measured
 *   as the densest thing in the ADT — 14 value registers, and 262 s_and_saveexec_b64 plus 351
 *   s_cbranch_execz across its inline copies, because a data-dependent walk compiles as divergent
 *   control flow. Adopting the array here retires that loop from the binding path.
 *   The environment today holds a SysPagedList per symbol (src_old/engine/environment.cuh:245); replacing
 *   it is the point of this file, not an incidental difference.
 *
 * WHY THE ELEMENT IS A WHOLE NODE, AND WHY A PACKED ONE WAS REFUSED.
 *   A packed {dtype, word, word} of 24 bytes was proposed and would have fit more values per chunk.
 *   It is refused because arc_drain walks a freed object as &heap[offset + k] — stride
 *   sizeof(sys__heap_node) — so a packed array would be strided past four cells in five and the cascade
 *   would miss most of the references it exists to release. Do not re-propose it without changing
 *   the collector first.
 *
 * WHERE THIS FILE LIVES AND HOW IT IS SPELLED.
 *   It sits in the sys folder and is written without a namespace, with a sys__ prefix, and every symbol
 *   it calls is spelled the same way: a global sys__heap__make, never a qualified name. The
 *   translation unit is closed around this tree — manifest.cu takes engine/manifest.cuh and pybind.cuh and
 *   nothing outside src — so there is no second spelling of any of these names in the binary for a
 *   qualifier to be needed against. A qualified call appearing here would be a call OUT of the tree.
 *
 * THE ALLOCATOR THIS FILE CALLS IS NOT THE ONE IN THE OLD TREE.
 *   The claim below calls the ported counter allocator. The old tree allocates chunks from a 64-bit
 *   bitmask and needs three steps with a window between them — claim the bit, retain the occupancy,
 *   retain the object — during which a chunk is allocated but uncounted. The compare-and-swap form
 *   merges the first two, which is why claiming here has two steps and not three.
 *   The two allocators never coexist in one binary: the bitmask's only callers are in the old engine
 *   directory, which the new tree replaces and does not include.
 *
 * PROVENANCE OF THE DESIGN.
 *   Architect, on the shape: "position 0 is the index, position 1 is the previous chunk ... so the
 *   index is LOCAL", and "if index is 1 then it means that we need to restore the previous chunk, and
 *   bump the index to 16".
 *   On the signature: "it transforms the signature into sys__stack__push(stack, value, arc)", and
 *   "the object is now owned by the stack so it is fine if this is the one keeping tabs".
 *   On growth: "the stack could grow in theory up until we finish the heap".
 *   On the public surface: "the only public methods should be push pop and peek", then "we do need 4
 *   methods after all, push pop peek and deallocate, where deallocating walks the chain and release
 *   the previous ones".
 *   On disposal: "we do not need to clear the things, after all we have the cursor for this reason",
 *   and on why the release is eager rather than left to the collector: "we dont want the lifetime of
 *   an object to depend on random chance of the detail of the length of the stack chunk".
 *   The counting rule this file follows — that containers account and opcodes do not — is an existing
 *   ruling, not a choice made here.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE CURSOR ───────────────────────────────────────────────────────────────────────────────────── */
/* How many values are in this chunk: one word out of its head, and nothing else.
 *
 * ⛳ NO CHUNK ANSWERS ZERO, which is the same answer an empty one gives, so nothing has to check before
 * asking. That is what lets the layers above read a top without first establishing there is a chunk to
 * read it from — and it is why an empty stack and a stack that never had a chunk are one state
 * everywhere, not just in `sys__stack__empty`.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint64_t sys__stack_chunk__zzprivate_cursor(const sys__stack_chunk* chunk) {
    if (chunk == 0) return 0ull;
    return sys__heap_node__head(chunk)->args[SYS__STACK_CHUNK__HEAD_CURSOR];
}

static __device__ inline void sys__stack_chunk__zzprivate_set_cursor(sys__stack_chunk* chunk, uint64_t n) {
    sys__heap_node__head(chunk)->args[SYS__STACK_CHUNK__HEAD_CURSOR] = n;
}

/* The chunk before this one, as an offset, and zero when there is none — which can mean that because the
 * heap allocation pool never hands out its first chunk, so nothing a stack could point back to sits at offset zero. */
static __device__ inline uint64_t sys__stack_chunk__zzprivate_previous(const sys__stack_chunk* chunk) {
    if (chunk == 0) return 0ull;
    return sys__heap_node__head(chunk)->args[SYS__STACK_CHUNK__HEAD_PREVIOUS];
}

static __device__ inline void sys__stack_chunk__zzprivate_set_previous(sys__stack_chunk* chunk, uint64_t offset) {
    sys__heap_node__head(chunk)->args[SYS__STACK_CHUNK__HEAD_PREVIOUS] = offset;
}

/* ── PEEK ─────────────────────────────────────────────────────────────────────────────────────────── */
/* Read the top value, and add a reference to it on the way out.
 *
 * The retain is done in the stack operations (which are the producer of the value), so the consumer can
 * just peek and then release. So looking is not free: what comes back is the caller's to hold and the
 * caller's to give up, exactly as if it had been popped. That is what makes it safe to keep — a peeked
 * value whose stack is then popped or released would otherwise be a pointer to something nobody owns.
 *
 * ⛔ AND IT COPIES OUT RATHER THAN HANDING BACK A POINTER, WHICH IS NOT A STYLE CHOICE.
 * The pointer would be wrong because an address on the array chunk is useless, what you want is the content.
 * Which is either a primitive object (opcode, int) or a reference to an object, so what you need is
 * to increment the counter of the holders of that original reference before handing it out.
 *
 * ⭐ AND THE COUNT THAT MOVES IS THE OBJECT'S, NOT THE CELL'S. What is copied out is the REFERENCE — a cell
 * whose args[0] names an object — and there is still exactly one object at the other end of it. Nothing is
 * duplicated; what changes is how many cells in the world name it, which is what the count in its head
 * says. Two references to one object are two names, not two things.
 *
 * ⛔ WHICH IS WHY THE CLAIM IS MADE HERE AND NOT BY THE COPY. A struct assignment in this subset is a
 * memcpy — there is no hook on it — so `sys__heap_node a = b;` duplicates a reference and counts nothing. The
 * obligation therefore lives in every function that HANDS A VALUE OUT or takes one in, and nowhere else:
 * peek and push take a reference, pop moves the one the stack had, and `sys__heap_object__set` exists
 * precisely because writing a cell over another one cannot be spelled as an assignment.
 *
 * An empty chunk answers with a null value rather than failing — asking what is on top of nothing is a
 * question with an answer, and a null IS that answer. If you really want to know the distinction check 
 * if the stack chunk reference is zero
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node sys__stack_chunk__zzprivate_peek(const sys__stack_chunk* chunk) {
    const uint64_t c = sys__stack_chunk__zzprivate_cursor(chunk);
    if (chunk == 0 || c == 0ull) return sys__heap_node__nothing();
    sys__heap_node out = chunk->cell[c - 1ull];
    if (sys__heap_node__carries_reference(out.dtype))
        (void)sys__heap_object__retain(out.args[0]);
    return out;
}


/* A chunk's own offset within the heap allocation pool, which is what the link stores and what the counts are kept
 * against. It reads the heap base address rather than being given it, because there is only one — see
 * `sys__heap__publish`. Masking cannot serve here: this crosses from address space into offset space,
 * and a mask only ever moves within one of them. */
static __device__ inline uint64_t sys__stack_chunk__zzpackage_offset(const sys__stack_chunk* chunk) {
    return sys__heap__offset((const sys__heap_node*)chunk);
}

/* ⛳ THERE IS NO LOCK AT THIS TIER, AND THAT IS NOT AN OMISSION.
 * The lock has to be on the stack holder, not on the stack chunks. The stack chunks are
 * internal by definition, so the access control is moved at the holder.
 *
 * A chunk is reachable only through the stack that owns it — there is no name for one anywhere else — so
 * holding the stack excludes everybody from every chunk in it. A second lock down here would be taken by
 * whoever already held the first, which is not exclusion, it is an ordering to get wrong.
 *
 * ⇒ The counting reaches chunks too, at teardown, and takes no lock either: it runs at a count of zero,
 * where by the package's own rule nobody can be holding one. */

/* ── CHUNK LIFETIME ──────────────────────────────────────────────────────────────────────────────── */
/* Put a freshly claimed chunk into its starting state: which chunk it follows, and a cursor of nothing.
 *
 * ⛳ IT INITIALISES, IT DOES NOT SET — the name matters because the two read the same at a call site and
 * mean opposite things about WHEN. This may be called once, before anything is in the chunk. Calling it
 * on a live one would wipe its cursor and orphan every value under it, so there is no version of this
 * that is safe to reach for later.
 *
 * ⛔ AND IT IS NOT OPTIONAL, WHICH IS THE WHOLE REASON IT EXISTS. A newly claimed chunk is not zeroed —
 * claiming moves a counter and never touches the nodes — so a recycled chunk arrives holding whatever its
 * last tenant left. A stale non-zero in the previous slot reads as a real predecessor, and the first pop
 * that emptied this chunk would hand back a chunk belonging to something else. The header is always
 * written, never assumed.
 *
 * Only the two header words need it. The value slots are never read above the cursor, so clearing them
 * would be writes spent making the chunk look tidy rather than be correct.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline void sys__stack_chunk__zzprivate_init(sys__stack_chunk* chunk,
                                                             const sys__stack_chunk* prev) {
    sys__stack_chunk__zzprivate_set_previous(chunk, (prev == 0) ? 0ull
                                                    : sys__stack_chunk__zzpackage_offset((const sys__stack_chunk*)prev));
    sys__stack_chunk__zzprivate_set_cursor(chunk, 0ull);
}

/* Take a granule, initialise it as the successor of `prev`, and return its address.
 *
 * ⛳ IT CARVES FROM THE ARRAY PROVIDER, NOT THE OBJECT ONE, WHICH IS THE ONLY DECISION HERE. A chunk goes
 * back to the heap allocation pool only when everything in it has died, so a chunk of things that churn must not also
 * hold things that last: the stack OBJECT is made from the other provider, and its chunks from this one.
 *
 * Null means the making failed, and the reason has already been raised where it was known — a caller has
 * nothing to add and nothing to guess. */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzprivate_create(const sys__stack_chunk* prev) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__STACK_CHUNK);
    if (head == 0) return (sys__stack_chunk*)0;
    sys__stack_chunk* fresh = (sys__stack_chunk*)(head + 1);
    sys__stack_chunk__zzprivate_init(fresh, prev);
    return fresh;
}

/* ── PUSH ─────────────────────────────────────────────────────────────────────────────────────────── */
/* Writes a value and returns the chunk that now holds the top, which the caller stores back into its
 * slot. The index needs no returning: the cursor lives in the chunk, so an address is a complete
 * answer.
 *
 * Push grows the stack itself. Running out of room is the only thing that changes the address, and it
 * is exactly the thing that needs an allocation, so a push that could not allocate would be unable to
 * answer its own question. Because it can, there is no "full" result to report and no null to return
 * for one.
 *
 * A null stack is the unbound state rather than an error. It is what a zeroed slot holds and what a
 * fully drained pop leaves behind, so "never bound" and "unbound again" have one representation, and
 * the first push and a growth are the same branch — which is also how a caller asks for a chunk at all:
 * pushing onto nothing claims one, writes its header, and returns it. There is no second way in.
 *
 * Because that representation is the only one, a chunk arriving here with a cursor of zero did not come
 * from these functions: a chunk exists only because a push put a value into it, and the pop that takes
 * the last value out hands the chunk back rather than leaving an empty one in the slot. So the guard is
 * a single equality and not a range. The two header words live in the head node, outside the cell array
 * entirely, so no cursor can name one — zero is the only reading these functions cannot produce, and it
 * is not a state to recover from.
 *
 * A null VALUE is a different matter and stays a fault. Making it mean "allocate without storing" would
 * give one argument two meanings, and a caller who simply forgot the value would get a silent chunk
 * instead of a complaint. Both of the two arguments this takes are required: a missing one is a fault
 * and never a mode.
 *
 * Push never returns null over a chunk it was given. Every refusal returns the chunk unchanged and
 * writes nothing, because the return value is going straight back into the structure's own
 * back-pointer, and a value that is both the result and the error cannot be assigned safely: a null
 * would orphan the whole chain, leaving every predecessor unreachable with its counts still held. The
 * symbol would simply read as unbound, which is a legal answer, so the leak would be silent.
 *
 * The accounting has three cells changing hands and only two of them are counted:
 *
 *   the pushed value   a second owner appears and the caller keeps its copy, so it is retained.
 *                      Missing this is the direction that frees a live object.
 *   a fresh chunk      retained once, for the slot about to hold it. Its allocator count is
 *                      already set by the claim.
 *   the old chunk      nothing. Its owner was that slot and is now the fresh chunk's previous — one
 *                      owner before, one after. Retaining it here would leak instead.
 *
 * ⛳ AND THE MEANS TO ALLOCATE IS AMBIENT RATHER THAN AN ARGUMENT: growth calls
 * `sys__heap__make`, which reads the provider off the kind's own row. There is nothing a caller
 * can leave off, so there is no push that grows while going uncounted. The one refusal here is the heap's
 * own — a make that fails hands back the chunk it was given, unchanged.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_transfer(sys__stack_chunk* chunk, const sys__heap_node* value) {
    /* ⛳ ONLY THE PER-CALL MISTAKE IS CHECKED HERE. A forgotten value is something this call can get wrong;
     * an unpublished heap base address and a block that never started are conditions of the launch, and they are
     * checked where they are established — by `sys__heap__make`, which is the only thing here that
     * can meet them and says which one it met: HPNP or HPNB, naming the cause whenever it lands — so
     * paying for the question on every push would buy nothing. */
    if (value == 0) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return chunk;                                    /* the chain is never clobbered */
    }
    /* The cursor is read once and used twice: to refuse a dead chunk, and to decide whether there is room.
     * Nothing is locked at this tier, and what arrives here does not all arrive under a lock either:
     * `sys__stack__push` holds the stack's, while `sys__stack__zzengine_compute_stack_push` and
     * `sys__stack__transfer` hold nothing at all, because the stacks they drive cannot be named by a
     * second block — the paragraphs over those two verbs carry that argument.
     * ⛔ WHICH MAKES A NEW CALLER A DECISION AND NOT AN ADDITION: an unlocked push racing a pop corrupts
     * the cursor, and a caller that hands over a stack somebody else can reach is what would do it. */

    const uint64_t cursor = sys__stack_chunk__zzprivate_cursor(chunk);
    if (chunk != 0 && cursor == 0ull) {                  /* a chunk in hand is never empty */
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return chunk;
    }

    uint64_t next;
    if (chunk == 0 || cursor >= (uint64_t)SYS__STACK_CHUNK__SLOTS) {
        sys__stack_chunk* fresh = sys__stack_chunk__zzprivate_create(chunk);       /* null stack: no predecessor */
        /* ⭐ AND THE ONE BEING LEFT IS NOT RELEASED HERE, WHICH IS WHAT THE THIRD ROW OF THE TABLE ABOVE
         * ARGUES: its owner was the caller's slot and is the fresh chunk's previous link the moment
         * `create` writes it — one owner before, one after. A release here would take a chunk the new one
         * points at to zero, and a retain would leak it; the count is right precisely because neither
         * happens. The fresh one needs no lock: it is not reachable from anywhere yet. */
        if (fresh == 0) return chunk;   /* make raised the reason; the chain is left as it was */
        fresh->cell[0] = *value;
        sys__stack_chunk__zzprivate_set_cursor(fresh, 1ull);
        return fresh;
    }
    next = cursor;

    chunk->cell[next] = *value;
    sys__stack_chunk__zzprivate_set_cursor(chunk, next + 1ull);
    return chunk;
}

/* ⭐ AND A PUSH IS A TRANSFER PLUS A HOLD, AND THE CODE SAYS SO. The two differ by exactly one thing —
 * whether the caller keeps what it put in — and writing
 * the difference as one line means they cannot drift apart, which two copies of the cursor arithmetic
 * eventually would. */
/* Did a transfer store the value? It did if it moved to a fresh chunk, or if the chunk in hand grew by
 * one. A transfer that refused — no value, a dead chunk, no room for a fresh one — hands back the chunk
 * it was given with its cursor where it was. */
static __device__ inline bool sys__stack_chunk__zzprivate_went_in(const sys__stack_chunk* was,
                                                                  uint64_t cursor_was,
                                                                  const sys__stack_chunk* now) {
    return now != was || (now != 0 && sys__stack_chunk__zzprivate_cursor(now) == cursor_was + 1ull);
}

static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_push(sys__stack_chunk* chunk,
                                                                        const sys__heap_node* value) {
    const uint64_t cursor_was = sys__stack_chunk__zzprivate_cursor(chunk);
    sys__stack_chunk* onto = sys__stack_chunk__zzpackage_transfer(chunk, value);
    /* ⛔ THE HOLD IS TAKEN ONLY FOR A VALUE THAT WENT IN. A refused transfer stored nothing, so a hold
     * taken for it would be one nothing ever gives back. */
    if (sys__stack_chunk__zzprivate_went_in(chunk, cursor_was, onto)
        && value != 0 && sys__heap_node__carries_reference(value->dtype))
        (void)sys__heap_object__retain(value->args[0]);
    return onto;
}

/* ── POP ──────────────────────────────────────────────────────────────────────────────────────────── */
/* Removes the top value and returns the chunk that is now current, which the caller stores back into
 * its slot exactly as it does for push. What it does NOT do is dispose of a chunk it emptied: that
 * chunk's offset goes back through `out_spent` and the caller owes it — the ⭐⭐ paragraph at the end of
 * the body has the reason, and the ordinary pop next door is the caller that pays it immediately.
 *
 * The value is copied into the caller's storage rather than returned by pointer, because this function
 * may release the very chunk the value sits in before it returns.
 *
 * ⭐ THE VALUE IS NOT RELEASED — IT IS TRANSFERRED, AND THAT IS THE WHOLE ACCOUNTING OF A POP.
 * This does not call release on the value as it is a simple transfer of ownership. 
 * The stack held one reference to it; the caller walks away with that same one. Nothing is
 * taken and nothing is given up, so the count does not move — which is exactly what a caller wants,
 * because it now has the value and nobody has released it out from under them.
 *
 * ⛳ IT IS THE COUNTERPART OF PEEK RATHER THAN OF PUSH. Peek leaves the stack holding its reference and
 * takes a NEW one for the caller; pop leaves the stack holding nothing and gives the caller the one it
 * had. Push is the other direction and takes its own, so a caller that pushes still owes a release on
 * what it pushed.
 * This is because as a lisp opcode we want the release to happen from the dereferencing of the object 
 * from the execution list, which happens automatically.
 *
 * ⛔ AND THE CURSOR GOES DOWN FIRST FOR THIS TO WORK. What empties a chunk walks the cells below the
 * cursor and lets go of each one; if the popped cell were still counted in it, the chunk going home would
 * release the very value just handed out.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_pop_deferring_disposal(
        sys__stack_chunk* chunk, sys__heap_node* out_value, uint64_t* out_spent) {
    if (out_spent != 0) *out_spent = 0ull;
    if (chunk == 0 || !sys__heap__zzpackage_published()) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return chunk;
    }

    const uint64_t c = sys__stack_chunk__zzprivate_cursor(chunk);
    if (c == 0ull) {                         /* nothing bound, or a cursor that cannot be produced */
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return chunk;
    }

    if (out_value != 0) *out_value = chunk->cell[c - 1ull];

    /* The cursor comes down before anything else, in both cases, so the cell just read is no longer part
     * of the chunk. That is what makes the transfer work: what empties a chunk walks only the cells below
     * the cursor, so the value handed out is not among them and nobody releases it. */
    sys__stack_chunk__zzprivate_set_cursor(chunk, c - 1ull);

    if (c > 1ull) {                                   /* values remain here — the common case */
        return chunk;
    }

    /* That was this chunk's last value. Whether the stack is finished or only this chunk is depends on
     * the link, read before the chunk is given up. */
    const uint64_t prev_offset = sys__stack_chunk__zzprivate_previous(chunk);
    sys__stack_chunk* prev = (prev_offset != 0ull) ? (sys__stack_chunk*)sys__heap__object_full_address(prev_offset) : (sys__stack_chunk*)0;

    /* ⛔ AND THE LINK IS CLEARED HERE, BECAUSE THE OWNERSHIP OF WHAT IT NAMES MOVES TO THE CALLER.
     * A move is a clear plus a write and never a retain: the predecessor is handed back below without its
     * count changing, so the reference the link held must stop existing at the same instant. Leaving it
     * for the collector would have the chunk released twice — once into the caller's hands and once out
     * of the link — and the second one takes a live stack to zero while somebody is holding it. */
    sys__stack_chunk__zzprivate_set_previous(chunk, 0ull);

    /* ⭐⭐ AND THE CHUNK IS NOT GIVEN UP HERE — ITS OFFSET IS HANDED BACK, AND THE CALLER OWES IT. The
     * cursor above is already zero and the link is cleared, so what is being handed over is an empty
     * chunk and nothing else; whoever takes it needs only to let go of the one hold it still carries.
     * ⛔ AND THE REASON IT IS NOT DONE HERE IS THE CALLER THAT CANNOT ALLOW IT. Giving anything up runs
     * the collector, and the collector drains a worklist — which it does by popping. A pop that gave a
     * chunk up would therefore be able to reach the very function that called it, and a routine that can
     * reach itself is a recursion as deep as the structure being torn down.
     * ⛳ THE ORDINARY POP NEXT DOOR DOES GIVE IT UP, so nothing outside the collector notices any of
     * this. The difference is only WHERE the giving-up happens. */
    if (out_spent != 0) *out_spent = sys__stack_chunk__zzpackage_offset(chunk);

    /* ⛔ THE PREDECESSOR RESUMES AT ITS OWN CURSOR, WHICH IS THE ONLY THING THAT KNOWS WHAT IT IS. A
     * chunk that filled and was left behind carries a full one, so reading it and assuming it give the
     * same answer — but only one of them stays right if a chunk ever becomes a predecessor by any
     * route other than filling. Writing over it would hand out cells nobody put there, silently: they
     * are real memory, they read as values, and nothing faults.
     *
     * Ownership of it moves from the link to the caller's slot, and a move is a clear plus a write, not a
     * retain — the release above hands the clearing to the collector.
     *
     * It is taken after this one is finished with rather than while it is held. Two at once is where an
     * order to get wrong comes from, and there is nothing here that needs both. */
    return prev;                                      /* null is the unbound state, and also the zero one */
}

/* The ordinary pop: the same thing, and then the emptied chunk goes. Every caller but the collector
 * wants this, which is why it keeps the plain name. */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_pop(sys__stack_chunk* chunk,
                                                                        sys__heap_node* out_value) {
    uint64_t spent = 0ull;
    sys__stack_chunk* next = sys__stack_chunk__zzpackage_pop_deferring_disposal(chunk, out_value, &spent);
    if (spent != 0ull) (void)sys__heap_object__release(spent);
    return next;
}

/* ── REPLACE ──────────────────────────────────────────────────────────────────────────────────────── */
/* Swap the top value for another one. The cursor does not move and the chain does not change, which is
 * why it is the one operation AT THIS TIER that hands nothing back: there is no address a caller could
 * need to store, and returning the stack would suggest it might have changed. Above, in the holder layer,
 * nothing hands a chunk back at all — the switch goes into the cell instead.
 *
 * It exists because doing the same thing with a pop and a push is not merely longer, it is expensive at
 * exactly the wrong moment. When the top is a chunk's only value, popping releases that chunk and hands
 * back its predecessor with a full cursor, so the push that follows finds no room and claims a fresh
 * chunk — a disposal, a queued object, a take off the free carousel and the compare-and-swap that
 * claims what came off it, to change one word. Replacing writes the word.
 *
 * The new value is retained before the old one is released, and the order is not defensive. Handing a
 * value back to the slot it already occupies is a legal thing for a caller to do, and releasing first
 * would take that object's count to zero, queue it for collection, and then retain what the collector has
 * already been promised.
 *
 * A stack with nothing in it is refused rather than treated as a push. Growing here would need the
 * allocation this exists to avoid, and a caller who meant to push has said something else.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline void sys__stack_chunk__zzprivate_replace(sys__stack_chunk* chunk, const sys__heap_node* v) {
    if (chunk == 0 || v == 0 || !sys__heap__zzpackage_published()) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return;
    }

    const uint64_t c = sys__stack_chunk__zzprivate_cursor(chunk);
    if (c == 0ull) {                              /* nothing bound here to replace */
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT);
        return;
    }

    /* The whole of it is under the lock, and it has to be: this reads which cell is the top, lets go of
     * what was in it, and writes the replacement. A reader arriving between the second and the third
     * would find a cell holding a value nobody owns any more. */
    sys__heap_object__set(&chunk->cell[c - 1ull], v);
}

/* ── A CHUNK PRETENDING TO RELEASE, AND WHY IT IS CORRECT ─────────────────────────────────────────────
 * The point of the release engine is to gather the things that a deallocating object was referencing
 * and decrement them by 1, and if that triggers the deallocation of any of those objects, return what 
 * needs deallocating as a stack of objects, so the release process has always a max depth of 2.
 * What this means in practical terms is that a stack, as a special case, will be dismantled by the
 * generic deallocation engine.
 * We can do this because each stack chunk is private and no public method allows for a retain count
 * bigger than one nor a direct reference to the internal values
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__stack_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack) {
    if (head == 0) return;
    sys__stack_chunk* chunk = (sys__stack_chunk*)(head + 1);

    /* Everything still standing in it moves out, and then its room goes back like anything else's — so a
     * chunk has ONE arm. Handing a chunk over whole instead would mean keeping
     * it alive to be read out of, which costs it a written-back count and a second death later; moving
     * its values out before this returns costs neither, and an object that is dead stays dead.
     *
     * ⛳ THE WALK IS BOUNDED BY THE CURSOR, so a cell above it is never read — those are cells a pop has
     * already handed to somebody, and reading one would move a value out twice. */
    const uint64_t cursor = sys__stack_chunk__zzprivate_cursor(chunk);
    for (uint64_t i = 0ull; i < cursor; ++i)
        sys__stack__transfer(releaser_stack, &chunk->cell[i]);

    /* And the chunk behind this one, which is an object like any other and is held by the link. */
    const uint64_t previous = sys__stack_chunk__zzprivate_previous(chunk);
    if (previous != 0ull) {
        sys__stack_chunk__zzprivate_set_previous(chunk, 0ull);
        sys__heap_node behind = sys__heap_object__reference_to(previous);
        sys__stack__transfer(releaser_stack, &behind);
    }
}

/* Returns the stack at an offset, or nothing as a fallback. What is checked here is what an offset can
 * answer for: that it could be one of ours, and that what it names says it is a stack. Anything else is
 * not a mistake to work around — it is a caller with the wrong value.
 *
 * ⭐ THE TAG THAT DECIDES IS IN THE HEAD, and reaching it is the ONLY reason the mask is here. The
 * address handed back is the one the offset already named, because an object always begins at the node
 * AFTER its head. */
static __device__ inline sys__heap_node* sys__stack__zzpackage_fields_at(uint64_t stack_offset) {
    if (stack_offset == 0ull) return (sys__heap_node*)0;
    if (!sys__heap_object__zzpackage_addressable(stack_offset)) return (sys__heap_node*)0;
    sys__heap_node* head = sys__heap_node__zzpackage_head(stack_offset);
    if (!sys__heap_object__is_type_from_head(head, SYS__KIND__STACK)) return (sys__heap_node*)0;
    return head + 1;
}

/* The same, for a caller holding a reference rather than an offset.
 *
 * ⭐ TWO TAGS, TWO QUESTIONS, AND THEY LIVE IN DIFFERENT PLACES — which is why this is a separate step
 * and not a wider argument on the one above. The tag in the caller's hand says this is A REFERENCE, i.e.
 * how to read `args[0]`: as a number, or as an offset to follow. The tag in the head says a reference TO
 * WHAT. Reading the first costs nothing, and only once it has been read is there an offset to hand on. */
static __device__ inline sys__heap_node* sys__stack__zzprivate_fields(const sys__heap_node* stack_reference) {
    if (stack_reference == 0 || stack_reference->dtype != SYS__KIND__OBJECT_REFERENCE) return (sys__heap_node*)0;
    return sys__stack__zzpackage_fields_at(stack_reference->args[0]);
}

static __device__ inline sys__stack_chunk* sys__stack__zzprivate_current_chunk(const sys__heap_node* stack_fields) {
    if (stack_fields == 0 || stack_fields->args[SYS__STACK__CURRENT_STACK_CHUNK] == 0ull) return (sys__stack_chunk*)0;
    return (sys__stack_chunk*)sys__heap__object_full_address(stack_fields->args[SYS__STACK__CURRENT_STACK_CHUNK]);
}

/* Write which chunk is the current one. The reference MOVES rather than being taken and given: growing
 * hands the old chunk to the new one's previous, shrinking hands it back, so the count a stack keeps on
 * the chunk it is working in is one from the first push to the last. */
static __device__ inline void sys__stack__zzprivate_set_current_chunk(sys__heap_node* stack_fields, sys__stack_chunk* chunk) {
    if (stack_fields == 0) return;
    stack_fields->args[SYS__STACK__CURRENT_STACK_CHUNK] = (chunk == 0) ? 0ull : sys__stack_chunk__zzpackage_offset(chunk);
}


/* Make a new stack, and hand back the reference that will be stored by the caller. */
/* What a program gets when it asks for a stack: the C constructor, with the parameters ignored because a
 * stack takes none. ⛳ THE WRAPPER IS THE WHOLE POINT — it is one line, it is where a kind's arguments
 * would be read if it had any, and it keeps the C verb below callable by anything not going through the
 * language. Neither knows about the other's caller. */
static __device__ inline sys__heap_node sys__stack__zzpackage_construct(sys__heap_node* base,
                                                                    const sys__heap_node* parameters) {
    (void)base; (void)parameters;
    return sys__stack__create();
}

static __device__ inline sys__heap_node sys__stack__create(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__STACK);
    if (head == 0) return sys__heap_node__nothing();   /* the making raised the reason */
    head[1].args[SYS__STACK__CURRENT_STACK_CHUNK] = 0ull;

    sys__heap_node reference;
    reference.dtype    = SYS__KIND__OBJECT_REFERENCE;
    reference.num_args = 0u;
    reference.args[0]  = sys__heap__offset(head + 1);
    return reference;
}

/* ⛳ IT ANSWERS WHETHER THE VALUE WENT IN. Every way it can fail has already raised its reason, so the
 * answer is for the caller that has to stop — one filling a list from a stack of work, say, would
 * otherwise carry on and hand back something with a piece missing. */
static __device__ inline bool sys__stack__push(const sys__heap_node* stack_reference, const sys__heap_node* value) {
    sys__heap_node* stack_fields = sys__stack__zzprivate_fields(stack_reference);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return false; }
    if (!sys__heap_object__lock(stack_reference->args[0])) return false;
    sys__stack_chunk* was = sys__stack__zzprivate_current_chunk(stack_fields);
    const uint64_t cursor_was = sys__stack_chunk__zzprivate_cursor(was);
    sys__stack_chunk* now = sys__stack_chunk__zzpackage_push(was, value);
    sys__stack__zzprivate_set_current_chunk(stack_fields, now);
    sys__heap_object__lock_release(stack_reference->args[0]);
    return sys__stack_chunk__zzprivate_went_in(was, cursor_was, now);
}

/* ── MOVING SOMETHING IN WITHOUT TAKING A HOLD OF IT ─────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"maybe what we need really is a transfer method in the stack, where it does not change
 * the arc count and allows us to pass the ownership from the whatever type object to the stack ... it
 * needs to be both on the chunk and on the main, because we need to retain the chunk address which might
 * otherwise change."*
 *
 * ⭐ SO THE HOLD DOES NOT MOVE, THE OWNER DOES. Whatever was keeping the value alive stops naming it and
 * this stack starts, and the count is the same on both sides of the line — which is why nothing here is
 * atomic and why there is no window in which the number is wrong.
 *
 * ⭐ AND IT IS ON THE STACK AND NOT ONLY ON THE CHUNK BECAUSE A TRANSFER CAN MOVE THE CHUNK. A full one
 * grows a fresh link and the address changes; a caller holding the old address would go on filling
 * something the stack has stopped pointing at. The stack keeps that address in its own field, so a
 * caller here holds an offset that stays true and never sees a chunk at all.
 *
 * ⛳ AND IT MAKES THE STACK IF THERE IS NOT ONE, which is what keeps a teardown that frees a leaf object
 * allocation-free. Most releases hand over nothing at all, and a stack made in advance for those would
 * be a chunk carved and given back on every one of them. The slot arrives holding nothing and the first
 * value is what turns it into a stack.
 *
 * ⭐⭐ AND IT TAKES NO LOCK, FOR THE REASON `sys__stack__zzpackage_empty_at` TAKES NONE: there is nobody
 * to exclude. Every caller hands it the slot a release chain is draining into, and that stack is made
 * here on the first value and ended by the chain that made it — its offset is never written anywhere
 * another block could read, so no second thread can name it. `push` above is the verb for a stack a
 * caller has published and it keeps its lock; this one is for a stack that has not been.
 * ⛳ THAT IS A REACHABILITY CLAIM — *"no second THREAD"*, not "no second block": the offset is never
 * written anywhere another reader could find it, and reachability does not care what the readers are.
 * ⛔ WHAT WOULD MAKE THAT FALSE, because it is a property of the CALLERS and not of this function: a
 * caller passing the offset of a stack that something else can reach. The lock cannot come back
 * quietly if that happens — a push racing a pop corrupts the cursor — so a new caller here is a
 * decision and not an addition. */

/* ── THE COMPUTE STACK'S OWN PUSH AND POP — NO LOCK, AND NO HOLD ────────────────────────────────────
 * ⚖ ARCHITECT: *"we cannot have a use after free because we are borrowing the hold of the list itself,
 * so we first pop the stack to get the reference and then we retrieve the list object, which means that
 * since this execution list is private we can borrow the engine's hold."*
 *
 * The evaluator's descent stack is the one stack in this tree that is BOTH uncontendable and unowning,
 * and the two facts are separate:
 *
 * ⭐ NO LOCK, FOR THE REASON `transfer` BELOW TAKES NONE: there is nobody to exclude. `places` lives
 *   inside one call of the evaluator, its offset is never written anywhere another block could read, and
 *   it is made and ended there. Each block that evaluates makes its own, so this stays true when more
 *   than one of them runs.
 * ⭐ NO HOLD, BECAUSE THE CHAIN ALREADY HAS ONE. An entry names a form the evaluator descended out of —
 *   and that form is still named by ITS parent's cell, all the way up to the program the engine holds.
 *   The stack borrows that hold rather than adding a second one, so a push costs no retain and a pop
 *   hands nothing over. **The cell naming a descended-into form is written only by the pop that returns
 *   to it**, which is the invariant this rests on and the one thing that would falsify it.
 *
 * ⛔ A NEW CALLER HERE IS A DECISION AND NOT AN ADDITION — the same sentence `transfer` carries, and for
 *   both halves: an unlocked push racing a pop corrupts the cursor, and an unowning stack outliving
 *   the chain that holds its entries is a use-after-free. Neither comes back quietly. */
/* ⛳ IT ANSWERS WHETHER THE PLACE WENT IN, like `sys__stack__push`. Every way it can fail has already
 * raised its reason; the answer is for the evaluator, which must stop rather than descend into a form
 * it has no way back out of. */
static __device__ inline bool sys__stack__zzengine_compute_stack_push(const sys__heap_node* stack_reference,
                                                                     const sys__heap_node* value) {
    sys__heap_node* stack_fields = sys__stack__zzprivate_fields(stack_reference);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return false; }
    sys__stack_chunk* was = sys__stack__zzprivate_current_chunk(stack_fields);
    const uint64_t cursor_was = sys__stack_chunk__zzprivate_cursor(was);
    sys__stack_chunk* now = sys__stack_chunk__zzpackage_transfer(was, value);
    sys__stack__zzprivate_set_current_chunk(stack_fields, now);
    return sys__stack_chunk__zzprivate_went_in(was, cursor_was, now);
}

static __device__ inline sys__heap_node sys__stack__zzengine_compute_stack_pop(const sys__heap_node* stack_reference) {
    if (stack_reference == 0 || stack_reference->dtype != SYS__KIND__OBJECT_REFERENCE) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing();
    }
    sys__heap_node* stack_fields = sys__stack__zzpackage_fields_at(stack_reference->args[0]);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing(); }
    sys__stack_chunk* chunk = sys__stack__zzprivate_current_chunk(stack_fields);
    if (chunk == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_EMPTY); return sys__heap_node__nothing(); }
    sys__heap_node out;
    uint64_t spent = 0ull;
    sys__stack__zzprivate_set_current_chunk(stack_fields,
        sys__stack_chunk__zzpackage_pop_deferring_disposal(chunk, &out, &spent));
    /* ⛳ THE ONE RELEASE THAT STAYS, AND IT IS NOT THE VALUE'S: `spent` is a CHUNK the pop emptied. The
     * stack owns its own room whatever it does or does not own of what is in it. */
    if (spent != 0ull) (void)sys__heap_object__release(spent);
    return out;
}

static __device__ inline void sys__stack__transfer(uint64_t* stack_offset, const sys__heap_node* value) {
    if (value == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return; }
    if (stack_offset == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return; }
    if (*stack_offset == 0ull) {
        const sys__heap_node made = sys__stack__create();
        if (made.dtype != SYS__KIND__OBJECT_REFERENCE) return;               /* the making raised */
        *stack_offset = made.args[0];
    }
    sys__heap_node* stack_fields = sys__stack__zzpackage_fields_at(*stack_offset);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return; }
    sys__stack__zzprivate_set_current_chunk(stack_fields,
        sys__stack_chunk__zzpackage_transfer(sys__stack__zzprivate_current_chunk(stack_fields), value));
}

static __device__ inline sys__heap_node sys__stack__zzpackage_pop_deferring_at(uint64_t stack_offset,
                                                                          uint64_t* out_spent) {
    if (out_spent != 0) *out_spent = 0ull;
    sys__heap_node* stack_fields = sys__stack__zzpackage_fields_at(stack_offset);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing(); }
    sys__stack_chunk* chunk = sys__stack__zzprivate_current_chunk(stack_fields);
    if (chunk == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_EMPTY); return sys__heap_node__nothing(); }
    if (!sys__heap_object__lock(stack_offset)) return sys__heap_node__nothing();
    sys__heap_node out;
    sys__stack__zzprivate_set_current_chunk(stack_fields,
        sys__stack_chunk__zzpackage_pop_deferring_disposal(chunk, &out, out_spent));
    sys__heap_object__lock_release(stack_offset);
    return out;
}

static __device__ inline sys__heap_node sys__stack__pop_deferring_disposal(const sys__heap_node* stack_reference,
                                                                       uint64_t* out_spent) {
    if (out_spent != 0) *out_spent = 0ull;
    if (stack_reference == 0 || stack_reference->dtype != SYS__KIND__OBJECT_REFERENCE) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing();
    }
    return sys__stack__zzpackage_pop_deferring_at(stack_reference->args[0], out_spent);
}

/* The ordinary pop, written in terms of the one above so there is one body and not two.
 * ⛳ AND THE EMPTIED CHUNK GOES AFTER THE LOCK IS BACK, WHICH IS WHERE THE NARROWEST HOLD PUTS IT.
 * Nothing about an empty, unlinked chunk touches the stack it came from, so keeping the stack held
 * across its disposal would buy nothing and would widen the stretch in which nobody else can pop. */
static __device__ inline sys__heap_node sys__stack__pop(const sys__heap_node* stack_reference) {
    uint64_t spent = 0ull;
    const sys__heap_node out = sys__stack__pop_deferring_disposal(stack_reference, &spent);
    if (spent != 0ull) (void)sys__heap_object__release(spent);
    return out;
}

/* Check if there is anything in it.
 * ⛳ EMPTY TAKES NO LOCK, AND THAT IS DELIBERATE RATHER THAN AN OVERSIGHT. ⚖ ARCHITECT: *"it is needed
 * in the measure that the deallocator knows when the stack is empty when it is doing the release chain.
 * so it is used only for a while check, for private stacks so you can do while (sys__stack__empty(stack))
 * ... so i think removing a lock is good and necessary."* On a stack nobody else can reach there is
 * nothing to exclude; on one somebody else can, a lock here would not help, because the answer is stale
 * the instant it is released and `while (!empty) pop` races either way. Asking and then acting is one
 * operation and this is not it. What it reads is a single aligned word, which cannot tear, so the answer
 * was true at some instant — which is all a caller can ever be given. */
static __device__ inline bool sys__stack__zzpackage_empty_at(uint64_t stack_offset) {
    sys__heap_node* stack_fields = sys__stack__zzpackage_fields_at(stack_offset);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return true; }
    return sys__stack__zzprivate_current_chunk(stack_fields) == 0;
}

static __device__ inline bool sys__stack__empty(const sys__heap_node* stack_reference) {
    if (stack_reference == 0 || stack_reference->dtype != SYS__KIND__OBJECT_REFERENCE) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return true;
    }
    return sys__stack__zzpackage_empty_at(stack_reference->args[0]);
}


/* Return the current top value of the stack without removing it.
 *⛔ READERS TAKE THE LOCK. A reader that does not lock is not
 * merely reading something mid-change: peek reads a cursor, then reads the cell under it, then takes a
 * reference to what it found — three steps, and a pop running between the first and the second can empty
 * the chunk, release it and hand its room back. The heap allocation pool does not zero what it hands out, so the cell
 * still reads as a plausible value and the retain lands on whatever offset those bytes contain.
 */
static __device__ inline sys__heap_node sys__stack__peek(const sys__heap_node* stack_reference) {
    sys__heap_node* stack_fields = sys__stack__zzprivate_fields(stack_reference);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing(); }
    if (!sys__heap_object__lock(stack_reference->args[0])) return sys__heap_node__nothing();
    const sys__heap_node out = sys__stack_chunk__zzprivate_peek(sys__stack__zzprivate_current_chunk(stack_fields));
    sys__heap_object__lock_release(stack_reference->args[0]);
    return out;
}

/* THE SAME READ, BORROWED — no lock taken and no hold handed out.
 *
 * ⛔⛔ IT IS ONLY CORRECT ON A CONTIGUOUS PIECE OF C AND THAT IS A READING OBLIGATION, NOT AN ENFORCED
 * ONE. ⚖ ARCHITECT: *"it is not something that we can trust a PROGRAM to have, it needs the discipline
 * of being a contiguous c codepath that makes sure there is no pop happening underneath in that stack or
 * we would be risking a use after free."* So the caller must take its own hold, or be finished with the
 * value, before anything can pop this stack.
 * ⛳ AND ITS FAILURE IS THE BAD KIND: a missing retain here is a USE AFTER FREE, where a missing release
 * elsewhere is a leak. It performs no event, so there is nothing to reconcile and no counter that would
 * notice — which is why it is marked and why nothing outside this package may call it.
 * ⛔ THE LOCK GOES WITH THE HOLD, AND ONLY BECAUSE NO OTHER RUNNER THREAD CAN REACH THIS STACK between
 * the read and the caller's own hold. A caller whose stack another block can name uses the ordinary
 * `peek`. */
static __device__ inline sys__heap_node sys__stack__zzpackage_peek_borrowed(const sys__heap_node* stack_reference) {
    sys__heap_node* stack_fields = sys__stack__zzprivate_fields(stack_reference);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return sys__heap_node__nothing(); }
    const sys__stack_chunk* chunk = sys__stack__zzprivate_current_chunk(stack_fields);
    if (chunk == 0) return sys__heap_node__nothing();
    const uint64_t c = sys__stack_chunk__zzprivate_cursor(chunk);
    if (c == 0ull) return sys__heap_node__nothing();
    return chunk->cell[c - 1ull];
}

/* Replace the current top value of the stack with a new value. Guaranteed to be churn-free
 * thanks to an in place swap of the value on the currently used array chunk
 */
static __device__ inline void sys__stack__replace(const sys__heap_node* stack_reference, const sys__heap_node* value) {
    sys__heap_node* stack_fields = sys__stack__zzprivate_fields(stack_reference);
    if (stack_fields == 0) { sys__fault__raise(0ull, SYS__STACK__FAULT_CONTRACT); return; }
    if (sys__stack__zzprivate_current_chunk(stack_fields) == 0) {
        sys__fault__raise(0ull, SYS__STACK__FAULT_EMPTY);
        return;
    }
    if (!sys__heap_object__lock(stack_reference->args[0])) return;
    sys__stack_chunk__zzprivate_replace(sys__stack__zzprivate_current_chunk(stack_fields), value);
    sys__heap_object__lock_release(stack_reference->args[0]);
}


/* ── A STACK PRETENDING TO RELEASE, AND WHY IT IS CORRECT ─────────────────────────────────────────────
 * The point of the release engine is to gather the things that a deallocating object was referencing
 * and decrement them by 1, and if that triggers the deallocation of any of those objects, return what 
 * needs deallocating as a stack of objects, so the release process has always a max depth of 2.
 * What this means in practical terms is that a stack, as a special case, will be dismantled by the
 * generic deallocation engine.
 * We can do this because each stack chunk is private and no public method allows for a retain count
 * bigger than one nor a direct reference to the internal values
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__stack__zzpackage_release_internal(sys__heap_node* head,
                                                                    uint64_t* releaser_stack) {
    if (head == 0) return;
    const uint64_t current = head[1].args[SYS__STACK__CURRENT_STACK_CHUNK];
    head[1].args[SYS__STACK__CURRENT_STACK_CHUNK] = 0ull;

    /* The clear is not only bookkeeping. This node's room goes back on the next line and the heap
     * allocation pool does not zero what it hands out, so anything still naming this dead stack reads a
     * zero rather than a chunk address that still looks live.
     *
     * ⭐ AND ONE MOVE IS THE WHOLE OF IT, WHICH IS WHAT MAKES A STACK CHEAP TO END. What it held is a
     * chunk, and a chunk is one object however many values are inside it — so this moves that one
     * reference and the chunk's own row deals with the values when its turn comes. */
    if (current == 0ull) return;
    sys__heap_node chain = sys__heap_object__reference_to(current);
    sys__stack__transfer(releaser_stack, &chain);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_STACK__IMPL_CUH */
