#ifndef SILVANN__PACKAGES_SYS_CPU_SYMBOL_CUH
#define SILVANN__PACKAGES_SYS_CPU_SYMBOL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"             /* how a refusal is said */
#include "heap_node__header.cuh"         /* the node every value in this language is made of */
#include "heap_object__header.cuh"       /* holds and references */
#include "heap__header.cuh"              /* what nothing is */
#include "string.cuh"                    /* a name is a string */
#include "dictionary.cuh"                /* and the tables it is registered in are dictionaries */
#include "system_register__header.cuh"   /* which is where the machine keeps them */
#include "../contracts/objects/symbol.cuh" /* its constants and fault words */
/* ══ symbols — a name registered once, numbered once, and never given back ═══════════════════════════
 *
 * ⚖ ARCHITECT: *"if i have a string i need to check if this string is already there or new, so i dont
 * add a duplicate symbol. this is the reason of hash2obj dictionary."* A program names things by number,
 * and the machine deals the numbers, from the text, so a name means one number on one machine however
 * many programs mention it and whichever composer built them.
 *
 * ── THE THREE TABLES ────────────────────────────────────────────────────────────────────────────────
 *     HASH_TO_STRING     a text's 512-bit hash → the one string holding that text
 *     STRING_TO_SYMBOL   a string's heap offset → its number, if it is a name at all
 *     SYMBOL_TO_STRING   a number → the string it was dealt for
 * Each is a register row naming a dictionary, and `sys`'s init stands all three up. ⚖ *"ok for the sys
 * init package. but i do wonder, should it be per block? no in this case it stays system wide."*
 * ⛳ A STRING'S OFFSET CAN BE A KEY BECAUSE A REGISTERED STRING NEVER DIES: the tables hold it for as
 * long as the machine is up, so its room is never handed to anything else.
 *
 * ── REGISTERING A NAME ──────────────────────────────────────────────────────────────────────────────
 * Hash the text and ask HASH_TO_STRING. Nothing there means the text is new: make its string and put it
 * in. Something there is compared byte for byte, and a mismatch is refused — at 512 bits two texts sharing
 * a hash is a broken hash, and a loud refusal is what keeps it from silently making two names one. Then ask
 * STRING_TO_SYMBOL. A number there is the answer; nothing there deals the next one and writes both ways.
 * ⭐ THE NEXT NUMBER IS SYMBOL_TO_STRING'S KEY COUNT. A dictionary keeps a key once it has one, so that
 * count only grows — which is what makes it safe to deal from without a counter of its own.
 * ⛔ AND A NUMBER CAN BE BURNED, WHICH IS HARMLESS. If the second write is refused, the first one's value
 * is taken back out; its key stays, so the count has moved past that number and nothing is ever dealt it.
 * A name asked for again gets the next number instead. `REASONED` from `sys__dictionary__remove`.
 *
 * ── WHAT IT DOES NOT DO ─────────────────────────────────────────────────────────────────────────────
 * It never gives a number back — the reason, and what giving one back would take, is at the top of
 * `bindings__header.cuh`. It takes no lock: the tables are written by `sys`'s init and then by whoever
 * registers names, which is the C door a composer uses, one launch at a time. It publishes no verb: a
 * program MENTIONS names, and the composer is what registers them.
 * ⛔ AND IT READS A ROW WITHOUT KEEPING ITS HOLD. The register's get hands back a hold, and each table is
 * given that hold straight back — `REASONED` safe on the premise that the register keeps its own hold on
 * every table for as long as the machine is up, which is true because nothing writes these rows again.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The dictionary a row read names, or nothing. It takes what the register's get answered — rows are only
 * ever named to the register's own three verbs — and gives the hold back; see the top of this file. */
static __device__ inline uint64_t sys__symbol__zzprivate_table(const sys__heap_node* held) {
    if (sys__heap_node__carries_reference(held->dtype)) (void)sys__heap_object__release(held->args[0]);
    if (held->dtype != SYS__KIND__OBJECT_REFERENCE || !sys__dictionary__is(held->args[0])) return 0ull;
    return held->args[0];
}

/* All three, or a refusal naming the missing table. */
static __device__ inline bool sys__symbol__zzprivate_tables(uint64_t* hashes, uint64_t* numbers,
                                                            uint64_t* names) {
    const sys__heap_node rows[3] = { sys__system_register__get(SYS__SYSTEM_REGISTER__HASH_TO_STRING),
                                     sys__system_register__get(SYS__SYSTEM_REGISTER__STRING_TO_SYMBOL),
                                     sys__system_register__get(SYS__SYSTEM_REGISTER__SYMBOL_TO_STRING) };
    *hashes  = sys__symbol__zzprivate_table(&rows[0]);
    *numbers = sys__symbol__zzprivate_table(&rows[1]);
    *names   = sys__symbol__zzprivate_table(&rows[2]);
    if (*hashes == 0ull || *numbers == 0ull || *names == 0ull) {
        sys__fault__raise(0ull, SYS__SYMBOL__FAULT_NO_TABLE);
        return false;
    }
    return true;
}

/* Stand the three tables up: an empty dictionary in each row, held by the register alone. `sys`'s init
 * calls it once per machine. Answers whether all three stand; a refusal has raised its own reason, and a
 * machine whose init refuses is taken back down, so rows written before the refusal go with it. */
static __device__ __noinline__ bool sys__symbol__zzpackage_stand_up(void) {
    uint64_t made[3] = { 0ull, 0ull, 0ull };
    bool placed = true;
    for (uint32_t i = 0u; i < 3u && placed; ++i) {
        made[i] = sys__dictionary__create();
        placed = made[i] != 0ull;                  /* the heap already raised */
    }
    placed = placed
          && sys__system_register__set(SYS__SYSTEM_REGISTER__HASH_TO_STRING,   SYS__KIND__OBJECT_REFERENCE, made[0])
          && sys__system_register__set(SYS__SYSTEM_REGISTER__STRING_TO_SYMBOL, SYS__KIND__OBJECT_REFERENCE, made[1])
          && sys__system_register__set(SYS__SYSTEM_REGISTER__SYMBOL_TO_STRING, SYS__KIND__OBJECT_REFERENCE, made[2]);
    /* The register's holds are the ones that stay; a table no row took goes here. */
    for (uint32_t i = 0u; i < 3u; ++i)
        if (made[i] != 0ull) (void)sys__heap_object__release(made[i]);
    return placed;
}

/* The number these bytes are known by on this machine, dealt the first time they are asked for — and, into
 * `interned` when it is not null, the string the table holds for them, WITHOUT a hold: that string lives as
 * long as the machine does, which is what lets a cell spell a name with it. A refusal answers
 * `SYS__SYMBOL__NONE` having raised, and leaves `interned` zero: text too long for one string or missing
 * its bytes, no tables, a hash collision, or no room.
 * ⛳ OUT OF LINE, because it hashes, looks up three times and may allocate — and it is called from a door
 * a composer crosses once per new name, never from inside a program. */
static __device__ __noinline__ uint64_t sys__symbol__intern(const uint8_t* bytes, uint64_t length,
                                                            uint64_t* interned) {
    if (interned != 0) *interned = 0ull;
    if (length > SYS__STRING__BYTES_MAX) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_LONG);
        return SYS__SYMBOL__NONE;
    }
    if (bytes == 0 && length != 0ull) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_NO_BYTES);
        return SYS__SYMBOL__NONE;
    }
    uint64_t hashes = 0ull, numbers = 0ull, names = 0ull;
    if (!sys__symbol__zzprivate_tables(&hashes, &numbers, &names)) return SYS__SYMBOL__NONE;

    /* ── THE STRING: the one already holding this text, or a new one. Either way this function holds it
     *    from here, and every path below gives that hold back. */
    sys__heap_node hash;
    sys__string__hash(bytes, length, &hash);
    uint64_t string = 0ull;
    const sys__heap_node known = sys__dictionary__get(hashes, &hash);
    if (known.dtype == SYS__KIND__OBJECT_REFERENCE) {
        string = known.args[0];
        if (!sys__string__holds(string, bytes, length)) {
            (void)sys__heap_object__release(string);
            sys__fault__raise(0ull, SYS__SYMBOL__FAULT_COLLISION);
            return SYS__SYMBOL__NONE;
        }
    } else {
        string = sys__string__create(bytes, length);
        if (string == 0ull) return SYS__SYMBOL__NONE;          /* the making already raised */
        const sys__heap_node reference = sys__heap_object__reference_to(string);
        if (!sys__dictionary__put(hashes, &hash, &reference)) {
            (void)sys__heap_object__release(string);
            return SYS__SYMBOL__NONE;
        }
    }

    /* ── THE NUMBER: the one it has, or the next one, written both ways. */
    const sys__heap_node by_string = sys__dictionary__number_key(string);
    const sys__heap_node numbered  = sys__dictionary__get(numbers, &by_string);
    uint64_t symbol = SYS__SYMBOL__NONE;
    if (numbered.dtype == SYS__KIND__VALUE_INT) {
        symbol = numbered.args[0];
    } else {
        const uint64_t next = sys__dictionary__count(names);
        const sys__heap_node by_symbol = sys__dictionary__number_key(next);
        const sys__heap_node reference = sys__heap_object__reference_to(string);
        if (sys__dictionary__put(names, &by_symbol, &reference)) {
            if (sys__dictionary__put(numbers, &by_string, &by_symbol)) symbol = next;
            else (void)sys__dictionary__remove(names, &by_symbol);   /* the number is burned — see above */
        }
    }
    /* ⛳ THE STRING IS HANDED OUT AFTER THIS FUNCTION'S HOLD IS GIVEN BACK, and that is sound: the hash table
     * took its own when the string went in, and nothing ever takes it out. */
    if (interned != 0 && symbol != SYS__SYMBOL__NONE) *interned = string;
    (void)sys__heap_object__release(string);
    return symbol;
}

/* Whether this is THE string the table holds for its own text — the one test that makes an uncounted
 * reference to it safe to put in a cell. A string anybody made is refused: it dies when its last hold
 * goes, and a name spelled with it would then name whatever the room holds next. Raises when it refuses.
 * ⛳ One hash of the text and one lookup, so it belongs where a program enters the machine, not on any
 * path a program runs. */
static __device__ __noinline__ bool sys__symbol__interned(uint64_t string) {
    uint64_t hashes = 0ull, numbers = 0ull, names = 0ull;
    if (!sys__symbol__zzprivate_tables(&hashes, &numbers, &names)) return false;
    bool same = false;
    if (sys__string__is(string)) {
        const sys__heap_node known = sys__dictionary__get(hashes, sys__string__hash_of(string));
        same = known.dtype == SYS__KIND__OBJECT_REFERENCE && known.args[0] == string;
        if (sys__heap_node__carries_reference(known.dtype)) (void)sys__heap_object__release(known.args[0]);
    }
    if (!same) sys__fault__raise(0ull, SYS__SYMBOL__FAULT_LOOSE);
    return same;
}

/* The string a number was dealt for, HELD — the caller gives that hold back. A number nothing was dealt
 * answers nothing, having raised. */
static __device__ inline sys__heap_node sys__symbol__name(uint64_t symbol) {
    uint64_t hashes = 0ull, numbers = 0ull, names = 0ull;
    if (!sys__symbol__zzprivate_tables(&hashes, &numbers, &names)) return sys__heap_node__nothing();
    const sys__heap_node key = sys__dictionary__number_key(symbol);
    const sys__heap_node string = sys__dictionary__get(names, &key);
    if (string.dtype != SYS__KIND__OBJECT_REFERENCE) {
        sys__fault__raise(0ull, SYS__SYMBOL__FAULT_UNKNOWN);
        return sys__heap_node__nothing();
    }
    return string;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SYMBOL_CUH */
