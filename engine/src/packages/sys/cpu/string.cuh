#ifndef SILVANN__PACKAGES_SYS_CPU_STRING_CUH
#define SILVANN__PACKAGES_SYS_CPU_STRING_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"       /* how a refusal is said */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh" /* the head every allocation begins with, and what it records */
#include "heap__header.cuh"        /* where the room comes from, and how wide a chunk is */
#include "../contracts/objects/string.cuh" /* its constants and fault words */
/* ══ a string — text the machine holds, rather than a number somebody made up for it ══════════════════
 *
 * ⚖ ARCHITECT: *"the dictionary was a detour because we wanted the init opcode of nn to reference let
 * bindings, and the let bindings should have been part of the json settings."* Settings arrive by NAME,
 * so a name has to be something the machine can hold and look up: its text, and not only a number.
 *
 * ── WHAT ONE IS MADE OF ─────────────────────────────────────────────────────────────────────────────
 *     the head   what every allocation begins with — the kind, the count, how many nodes it reaches —
 *                and in `args[1]` the LENGTH IN BYTES, which a count of nodes cannot say
 *     node 1     the HASH, all sixty-four bytes of it — and it is the node a reference names
 *     node 2 …   the TEXT, sixty-four bytes to a node, the last one padded out with zeros
 * ⚖ ARCHITECT: *"the byte length is an argument of the head heap node, and the hash is the full second
 * node otherwise we would have to hash at 48 bytes which looks inefficient."*
 * ⛳ `args[0]` IS NOT FREE, WHICH IS WHY THE LENGTH IS ONE ALONG: it is the object lock in every kind
 * (`SYS__HEAP_OBJECT__LOCK`). A list chunk keeps its own two fields in the same place for the same reason.
 * ⭐ AND THE HASH BEING WHAT A REFERENCE NAMES IS NOT AN ACCIDENT OF THE LAYOUT. A reference names the node
 * after the head in every kind; here that node is the thing a lookup compares, so the first load a caller
 * pays is the one it needed.
 * ⛔ THE TEXT IS BYTES AND NOT CELLS. Nothing in it carries a kind, so nothing may step through it asking
 * one — which nothing does, because a string holds no references and its row hands nothing over.
 *
 * ── WHY THE HASH IS FIVE HUNDRED AND TWELVE BITS ────────────────────────────────────────────────────
 * At that width two different texts sharing a hash is not something that happens by chance: across a
 * trillion names the odds are about one in 10^130. So equal hashes can be read as equal text, and a
 * lookup compares eight words and never the bytes.
 * `REASONED`, from the birthday bound; the premise is BLAKE2b's collision resistance, which is published
 * cryptography rather than a property this tree would have to take on trust.
 * ⛔ WHAT MAKES TRUSTING IT SAFE IS THE ONE PLACE THAT DOES NOT: whoever registers a name compares the
 * bytes once when a hash is already known, and refuses on a mismatch — so a broken hash is a loud refusal
 * rather than two names silently becoming one. `sys__string__holds` is that comparison; `sys__string__same`
 * compares two strings the same way.
 *
 * ── WHY BLAKE2b ─────────────────────────────────────────────────────────────────────────────────────
 * It works in 64-bit words, it needs no table but its message schedule, and its reference ships with
 * Python (`hashlib.blake2b`) — so the host harness checks this code against it byte for byte rather than
 * against a figure somebody copied. What it costs does not matter here: a string is hashed once, when it
 * is made, and never while a program runs.
 * ⚖ *"blake2b could be a c function itself or an opcode too, it will be noinlined anyway."* It is a C
 * function. ⛳ A VERB IS ONE ROW AWAY AND IS NOT WRITTEN, because it has nowhere to put its answer: five
 * hundred and twelve bits do not fit the forty-eight a cell carries, so a program asking for a hash would
 * need an object to hold one, and no program has asked.
 *
 * ── IT IS COUNTED LIKE ANY OBJECT, AND THAT COSTS NOTHING ───────────────────────────────────────────
 * Taking a hold never looks at the kind — it adds one to the word in the head — so counting a string is
 * free, and a string with no counting would have put a kind test on every hold anybody takes on anything.
 * What is empty is its RELEASE: a string holds nothing, so when its last hold goes it owes its room and no
 * more, which is the default row.
 * ⛳ A STRING THAT IS NEVER FREED IS A HOLDER'S DECISION, NOT THE KIND'S. The table that registers names
 * keeps its hold for as long as the machine is up; a string a program makes and does not register goes
 * when its last holder lets go, like anything else.
 *
 * ── THE LONGEST ONE THERE CAN BE ────────────────────────────────────────────────────────────────────
 * A string is one allocation, and one allocation is at most one chunk less the chunk's own head — so at
 * the shipping width 510 nodes, of which the head and the hash take two and the text gets 508: 32,512
 * bytes. `REASONED` from `sys__heap__zzpackage_make_sized` and `SYS__CHUNK__FIRST`.
 * ⚖ *"the max length of a string is 509 nodes x 64bytes"* — 509 is everything after the head, and the
 * hash is one of them.
 * ⛔ THE LIMIT IS CHECKED HERE, BEFORE ANY ROOM IS ASKED FOR. The allocator would refuse the same request,
 * but only after claiming a fresh chunk to find out it was too small.
 *
 * ── WHAT IT DOES NOT DO ─────────────────────────────────────────────────────────────────────────────
 * It does not look for an existing string with the same text: two makes of one text are two objects, and
 * keeping one per text is the dictionary's job, not this file's. It is never written after it is made,
 * so reading one somebody holds needs no lock. It cannot be copied (its row refuses, as every row does
 * today) and a program cannot build one (`create` refuses — strings are made when a program is composed,
 * through the C door, not by the program).
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE HASH ────────────────────────────────────────────────────────────────────────────────────────
 * BLAKE2b with no key and a 64-byte digest, as RFC 7693 states it. The words are read little-endian and
 * the digest is written the same way, so the node holds exactly the bytes `hashlib.blake2b` answers.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

static __device__ inline uint64_t sys__string__zzprivate_rotate(uint64_t word, uint32_t by) {
    return (word >> by) | (word << (64u - by));
}

/* One quarter-round, over four of the sixteen working words and two words of the message. */
static __device__ inline void sys__string__zzprivate_mix(uint64_t* v, uint32_t a, uint32_t b, uint32_t c,
                                                         uint32_t d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x;  v[d] = sys__string__zzprivate_rotate(v[d] ^ v[a], 32u);
    v[c] = v[c] + v[d];      v[b] = sys__string__zzprivate_rotate(v[b] ^ v[c], 24u);
    v[a] = v[a] + v[b] + y;  v[d] = sys__string__zzprivate_rotate(v[d] ^ v[a], 16u);
    v[c] = v[c] + v[d];      v[b] = sys__string__zzprivate_rotate(v[b] ^ v[c], 63u);
}

/* Fold one 128-byte block into the state. The block is read straight out of the text, starting at `from`,
 * and every byte at or past `length` reads as zero — which is the padding the last block wants, without a
 * copy of it.
 * ⛳ NO LOCAL BUFFER FOR THE BLOCK, and that is the frame and not tidiness: a callee's frame is paid by
 * whatever calls it, and this can be reached from inside the evaluator once a package looks a setting up
 * by name. The message words and the working words are what it keeps, and nothing else.
 * ⛳ OUT OF LINE, FOR SIZE AND NOT FOR SPEED: inlined, its two call sites were two copies of twelve rounds.
 * `REASONED` on the host — the timing was measured only on the device kernel (▶ the note at the top). */
static __device__ __noinline__ void sys__string__zzprivate_compress(uint64_t* h, const uint64_t* iv,
                                                              const uint8_t* bytes, uint64_t from,
                                                              uint64_t length, uint64_t counted, bool last) {
    static const uint8_t schedule[10][16] = {
        {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
        { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
        { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
        {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
        {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
        {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
        { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
        { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
        {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
        { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    };
    uint64_t m[16];
    for (uint32_t i = 0u; i < 16u; ++i) {
        uint64_t word = 0ull;
        for (uint32_t k = 0u; k < 8u; ++k) {
            const uint64_t at = from + (uint64_t)(i * 8u + k);
            if (at < length) word |= (uint64_t)bytes[at] << (8u * k);
        }
        m[i] = word;
    }
    uint64_t v[16];
    for (uint32_t i = 0u; i < 8u; ++i) { v[i] = h[i]; v[i + 8u] = iv[i]; }
    /* The byte counter is 128 bits and a string is at most a chunk, so its high half is always zero. */
    v[12] ^= counted;
    if (last) v[14] = ~v[14];
    for (uint32_t r = 0u; r < 12u; ++r) {
        const uint8_t* s = schedule[r % 10u];
        sys__string__zzprivate_mix(v, 0u, 4u,  8u, 12u, m[s[0]],  m[s[1]]);
        sys__string__zzprivate_mix(v, 1u, 5u,  9u, 13u, m[s[2]],  m[s[3]]);
        sys__string__zzprivate_mix(v, 2u, 6u, 10u, 14u, m[s[4]],  m[s[5]]);
        sys__string__zzprivate_mix(v, 3u, 7u, 11u, 15u, m[s[6]],  m[s[7]]);
        sys__string__zzprivate_mix(v, 0u, 5u, 10u, 15u, m[s[8]],  m[s[9]]);
        sys__string__zzprivate_mix(v, 1u, 6u, 11u, 12u, m[s[10]], m[s[11]]);
        sys__string__zzprivate_mix(v, 2u, 7u,  8u, 13u, m[s[12]], m[s[13]]);
        sys__string__zzprivate_mix(v, 3u, 4u,  9u, 14u, m[s[14]], m[s[15]]);
    }
    for (uint32_t i = 0u; i < 8u; ++i) h[i] ^= v[i] ^ v[i + 8u];
}

/* Hash `length` bytes into the sixty-four bytes of `into`. The bytes may be absent only when there are
 * none; `into` is written whole, and nothing else is.
 * ⛳ THE LAST BLOCK IS NEVER EMPTY UNLESS THE TEXT IS: a text whose length is a multiple of 128 finishes
 * on its own last full block, which is what the standard says and what the harness pins at 128 and 256. */
static __device__ __noinline__ void sys__string__hash(const uint8_t* bytes, uint64_t length,
                                                      sys__heap_node* into) {
    if (into == 0) return;
    static const uint64_t iv[8] = {
        0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
        0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
    };
    uint64_t h[8];
    for (uint32_t i = 0u; i < 8u; ++i) h[i] = iv[i];
    h[0] ^= 0x01010040ull;             /* the parameter block: no key, a digest of sixty-four bytes */
    uint64_t from = 0ull;
    while (length - from > 128ull) {
        sys__string__zzprivate_compress(h, iv, bytes, from, length, from + 128ull, false);
        from += 128ull;
    }
    sys__string__zzprivate_compress(h, iv, bytes, from, length, length, true);
    /* ⛳ WRITTEN A BYTE AT A TIME, because a node is a struct and a byte is the one type allowed to name any
     * object's storage. Reading it back as words is the caller's business and the layout makes it exact. */
    unsigned char* out = (unsigned char*)into;
    for (uint32_t i = 0u; i < 8u; ++i)
        for (uint32_t k = 0u; k < 8u; ++k) out[i * 8u + k] = (unsigned char)(h[i] >> (8u * k));
}

/* ── MAKING ONE, AND ASKING WHAT IT HOLDS ────────────────────────────────────────────────────────────
 * The caller of `create` holds what comes back and gives it up like any other reference. Everything else
 * here reads, takes nothing and owes nothing.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Make one holding these `length` bytes and answer the reference, which names its hash. Refused, with a
 * raise and a zero, when the text is longer than one string can hold or the bytes are missing; an empty
 * text is a string like any other and needs no bytes. */
static __device__ inline uint64_t sys__string__create(const uint8_t* bytes, uint64_t length) {
    if (length > SYS__STRING__BYTES_MAX) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_LONG);
        return 0ull;
    }
    if (bytes == 0 && length != 0ull) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_NO_BYTES);
        return 0ull;
    }
    const uint64_t node = (uint64_t)sizeof(sys__heap_node);
    const uint64_t text_nodes = (length + node - 1ull) / node;
    sys__heap_node* head = sys__heap__zzpackage_make_sized(SYS__KIND__STRING,
                                                           (uint64_t)SYS__STRING__NODES_MIN + text_nodes);
    if (head == 0) return 0ull;                    /* the heap already raised */
    head->args[SYS__STRING__LENGTH] = length;
    sys__string__hash(bytes, length, head + 1);
    /* ⛔ THE PAD IS WRITTEN, NOT ASSUMED. A claimed chunk is not zeroed, so the tail of the last node holds
     * whatever the last tenant left, and a reader comparing whole nodes would see it. */
    unsigned char* text = (unsigned char*)(head + 2);
    for (uint64_t i = 0ull; i < text_nodes * node; ++i) text[i] = (i < length) ? bytes[i] : (unsigned char)0;
    return sys__heap__offset(head + 1);
}

/* Whether this names one at all. Asked by every reader below before it reaches. */
static __device__ inline bool sys__string__is(uint64_t string) {
    if (!sys__heap_object__zzpackage_addressable(string)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(string), SYS__KIND__STRING);
}

/* How many bytes of text. ⛔ AN EMPTY STRING AND A REFUSAL BOTH ANSWER ZERO — the refusal also raises, and
 * a caller that must tell them apart asks `is` first. */
static __device__ inline uint64_t sys__string__length(uint64_t string) {
    if (!sys__string__is(string)) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_KIND);
        return 0ull;
    }
    return sys__heap_node__zzpackage_head(string)->args[SYS__STRING__LENGTH];
}

/* Where the text starts, good for `length` bytes and for as long as the caller holds the string. */
static __device__ inline const uint8_t* sys__string__bytes(uint64_t string) {
    if (!sys__string__is(string)) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_KIND);
        return (const uint8_t*)0;
    }
    return (const uint8_t*)(sys__heap__object_full_address(string) + 1);
}

/* The node holding its hash — which is the node the reference names, asked for by a name that says so. */
static __device__ inline const sys__heap_node* sys__string__hash_of(uint64_t string) {
    if (!sys__string__is(string)) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_KIND);
        return (const sys__heap_node*)0;
    }
    return sys__heap__object_full_address(string);
}

/* Whether a string holds exactly these bytes — the check a name registered by its hash makes against the
 * string that hash already names. Something that is not a string holds nothing, and says so. */
static __device__ inline bool sys__string__holds(uint64_t string, const uint8_t* bytes, uint64_t length) {
    if (!sys__string__is(string)) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_KIND);
        return false;
    }
    if (sys__string__length(string) != length) return false;
    if (bytes == 0 && length != 0ull) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_NO_BYTES);
        return false;
    }
    const uint8_t* text = sys__string__bytes(string);
    for (uint64_t i = 0ull; i < length; ++i) if (text[i] != bytes[i]) return false;
    return true;
}

/* Whether two strings hold the same TEXT — the bytes, not the hashes. It is the one comparison that does
 * not trust the hash, which is what lets every other one trust it (see the top of this file). Something
 * that is not a string is never the same as anything, and says so. */
static __device__ inline bool sys__string__same(uint64_t a, uint64_t b) {
    if (!sys__string__is(a) || !sys__string__is(b)) {
        sys__fault__raise(0ull, SYS__STRING__FAULT_KIND);
        return false;
    }
    if (a == b) return true;
    const uint64_t length = sys__string__length(a);
    if (length != sys__string__length(b)) return false;
    const uint8_t* x = sys__string__bytes(a);
    const uint8_t* y = sys__string__bytes(b);
    for (uint64_t i = 0ull; i < length; ++i) if (x[i] != y[i]) return false;
    return true;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_STRING_CUH */
