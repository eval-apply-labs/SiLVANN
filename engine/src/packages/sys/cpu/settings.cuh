#ifndef SILVANN__PACKAGES_SYS_CPU_SETTINGS_CUH
#define SILVANN__PACKAGES_SYS_CPU_SETTINGS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"       /* how a refusal is said */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh" /* holds and references */
#include "string.cuh"                  /* the file arrives as one, and every value becomes one */
#include "dictionary.cuh"              /* and what the parse answers is one */
#include "system_register__header.cuh" /* and the row the boot leaves it in */
#include "node_array__header.cuh"    /* one dictionary a worker, in an array */
#include "../contracts/objects/settings.cuh" /* its constants and fault words */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * ⚖ *"sys and the evaluator go host"* takes away the register argument for the host/card split below
 * (*"a parser on the card costs registers on the hot path's own kernel"*) without taking away the split:
 * on a CPU there is no such kernel. The split survives on ownership. Do not repeat the register
 * argument; it is false.
 * ⛳ the file's "on the card" became "inside the engine" — both readers run on the CPU — and
 *   no register sentence is left outside this block.
 * ⛳ RETIREMENT: when the evaluator runs on the host and the register sentence has been dropped — both
 *   true so this block is ready to go.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ the settings — a config file, read inside the engine, answered as a sealed dictionary ═══════════
 *
 * ⚖ ARCHITECT: *"the settings are a sealed dictionary of key-hash -> value string, not let bindings, so
 * we dont pollute the bindings for startup variables."* And on where the parse happens: *"having the
 * element sending the string internally and config2dict to process it is perfectly fine"* — one string
 * in, one dictionary out, parsed HERE rather than split on the host, because splitting it there would
 * put a second implementation of this format in Python.
 *
 * ── THE FORMAT, RULED ───────────────────────────────────────────────────────────────────────────────
 *     `package__object__key:value`, one per line
 *     `\n` ends a line · `\r` before it is ignored · split at the FIRST colon
 *     a space, or anything else weird, is refused — as is an empty key, an empty value, and a line
 *     with no colon at all
 * ⚖ *"the settings file is key:value lines parsed in C (refuse spaces/anything weird), not JSON."*
 *
 * ── WHAT IT ANSWERS ─────────────────────────────────────────────────────────────────────────────────
 * A SEALED dictionary: the key is the HASH OF THE KEY TEXT — the same 512-bit node `sys__symbol` keys
 * its tables by — and the value is a STRING holding the text after the colon. The key text itself is
 * not kept: a reader has the name it is looking for, so it can hash it and ask.
 * ⚖ THE SEAL IS WHAT MAKES IT SAFE TO PUBLISH, NOT THE HIDING: a sealed dictionary is read with no lock,
 * which is what the async compute needs anyway, and the row that names it stays writable like every
 * other row — *"it might even be correct to have it modifiable, if someone wants to load a new config."*
 * ⚖ AND THE FILE'S OWN STRING IS THE BOOT'S TO RELEASE once this answers: the keys became hashes and the
 * values became strings of their own, so nothing here keeps a pointer into the original text.
 *
 * ⛳ NO CONFIG FILE STILL WRITES ONE, EMPTY — the boot calls this with an empty string rather than
 * skipping it, so an init has ONE failure mode (the key is not there) instead of two.
 *
 * ── WHY THE PARSE ALLOCATES NOTHING OF ITS OWN ──────────────────────────────────────────────────────
 * It never copies a line out. A line is a pair of offsets into the string's own bytes, the colon is
 * found inside it, and the key is hashed and the value made straight from that range — so there is no
 * line buffer, no bound on a line's length beyond the string's own, and nothing to get wrong about
 * where a copy ends. ⛳ That is also why `\r` costs nothing: it is not printable, so the byte check
 * would refuse it anywhere, and the ONE place it is legitimate — immediately before the `\n` of a file
 * written on Windows — is a single decrement of the line's end.
 *
 * ⛔⛔ FOUR THINGS HERE WERE NOT RULED, AND ALL FOUR ARE DECIDED IN THE REFUSING DIRECTION ON PURPOSE.
 * A parser that is later RELAXED never changes the meaning of a file that already worked; one that is
 * later TIGHTENED breaks files that did. So where the ruling was silent this refuses, and every one of
 * them is a one-line flip if the architect wants it the other way:
 *     ① A BLANK LINE IS REFUSED, except the empty one after the file's last `\n`. A file ending in a
 *        newline has to work — every text editor writes one — but a blank line BETWEEN entries is
 *        "anything else weird" until somebody says otherwise. ▶ `zzprivate_line`'s caller.
 *     ② A BYTE MUST BE PRINTABLE ASCII AND NOT A SPACE — 0x21..0x7E. That is the strictest reading of
 *        *"refuse spaces/anything weird"*, and it makes a tab, a stray control byte and any UTF-8
 *        sequence all refusals rather than three different silent outcomes. ⛳ A value that wants a
 *        space is the first real request that would flip this.
 *     ③ A DUPLICATE KEY IS REFUSED rather than replaced. `put` would happily overwrite, and a config
 *        naming one key twice is a mistake whose cost is an evening spent reading the wrong value.
 *     ④ THE `package__object__key` SHAPE IS NOT ENFORCED — it is the code's convention for how a name
 *        is spelled, and this parser is not the thing that should learn every package's spelling. A key
 *        is text with no colon and no space in it; what it MEANS is the asking init's business.
 * `REASONED`, and the premise is that the strict direction is the recoverable one. None of the four is
 * observable from outside a refusal message, so flipping any of them breaks nothing that reads a value.
 *
 * ⛳⛳ AND THE NAME IS THE ONE THING TO RULE BEFORE THIS IS BUILT ON. The ruling wrote
 * `sys__dictionary__from_string`, and that name cannot live where it says: `dictionary.cuh` is included
 * BEFORE `string.cuh`, so a function in it cannot call `sys__string__create`. Beyond the mechanics, this
 * tree names a component for its CONCERN and not for what it builds — `sys__symbol__intern` makes and
 * reads three dictionaries and is not called `dictionary__something`. So the parser is filed here as
 * `sys__settings__from_string`. ⛔ IT IS A `sed` AWAY FROM THE RULED SPELLING and nothing else depends on
 * which it is; ▶ the RESUME banner, which asks.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The one byte test, and it is the whole of *"refuse spaces/anything weird"*: printable ASCII, space
 * excluded. A colon passes — it is printable — which is what lets a VALUE hold one; only the FIRST colon
 * in a line is the split, so `a__b__c:http://x` reads its value whole. */
static __device__ inline bool sys__settings__zzprivate_plain(uint8_t byte) {
    return byte >= 0x21u && byte <= 0x7Eu;
}

/* One line, already cut at its `\n` and with a `\r` taken off its end: check it, hash its key, make its
 * value and enter the pair. Answers whether it happened; every refusal has already raised.
 * ⛔ THE CALLER STILL HOLDS THE DICTIONARY — this takes nothing and leaves nothing behind on a refusal:
 * the value string it makes is released here the moment the dictionary has a hold of its own, and it is
 * released WITHOUT being entered if the entry is refused.
 * ⛳ OUT OF LINE, for the reason `intern` is: it hashes, looks up, allocates and may rotate a tree, and
 * it is reached once per line at boot and from nowhere a program can run. */
static __device__ __noinline__ bool sys__settings__zzprivate_line(uint64_t dictionary,
                                                                  const uint8_t* bytes,
                                                                  uint64_t from, uint64_t to) {
    /* ── the colon, and every byte around it ─────────────────────────────────────────────────────── */
    uint64_t colon = to;
    for (uint64_t i = from; i < to; ++i) {
        if (!sys__settings__zzprivate_plain(bytes[i])) {
            sys__fault__raise(0ull, SYS__SETTINGS__FAULT_BYTE);
            return false;
        }
        if (bytes[i] == (uint8_t)':' && colon == to) colon = i;
    }
    if (colon == to) {
        sys__fault__raise(0ull, SYS__SETTINGS__FAULT_LINE);
        return false;
    }
    /* ⛔ BOTH HALVES, SEPARATELY. `:value` and `key:` are different mistakes from `key` with no colon,
     * and a reader of the log should not have to guess which of the three a file made. */
    if (colon == from || colon + 1ull == to) {
        sys__fault__raise(0ull, SYS__SETTINGS__FAULT_EMPTY);
        return false;
    }

    /* ── the key, which is kept only as its hash ─────────────────────────────────────────────────── */
    sys__heap_node key;
    sys__string__hash(bytes + from, colon - from, &key);

    /* ⛔ ASKED BEFORE IT IS WRITTEN, because `put` REPLACES and a replacement is silent. The answer is
     * held when there is one, so it is given back before the refusal. */
    const sys__heap_node standing = sys__dictionary__get(dictionary, &key);
    if (standing.dtype != SYS__KIND__VALUE_NULL) {
        if (sys__heap_node__carries_reference(standing.dtype)) {
            (void)sys__heap_object__release(standing.args[0]);
        }
        sys__fault__raise(0ull, SYS__SETTINGS__FAULT_TWICE);
        return false;
    }

    /* ── the value, which becomes a string of its own ────────────────────────────────────────────── */
    const uint64_t value = sys__string__create(bytes + colon + 1ull, to - colon - 1ull);
    if (value == 0ull) return false;               /* the making already raised */
    const sys__heap_node reference = sys__heap_object__reference_to(value);
    const bool entered = sys__dictionary__put(dictionary, &key, &reference);
    /* Either way this function's own hold goes back: the dictionary took one of its own when it entered,
     * and when it refused there is nobody left to want this. */
    (void)sys__heap_object__release(value);
    return entered;
}

/* Read the whole config and answer a SEALED dictionary, held by the caller. Zero, having raised, for
 * anything that is not a string and for the first line that will not parse.
 *
 * ⛔ A REFUSAL LEAVES NOTHING. The dictionary is released on the way out, which hands back every value
 * already entered — so a half-read config never reaches a register row, and an init cannot be handed
 * settings that stop halfway through the file.
 * ⛳ AN EMPTY STRING ANSWERS AN EMPTY SEALED DICTIONARY, which is the no-config-file case and is why
 * that case needs no branch anywhere else. */
static __device__ inline uint64_t sys__settings__from_string(uint64_t text) {
    if (!sys__string__is(text)) {
        sys__fault__raise(0ull, SYS__SETTINGS__FAULT_KIND);
        return 0ull;
    }
    const uint8_t* bytes  = sys__string__bytes(text);
    const uint64_t length = sys__string__length(text);

    const uint64_t dictionary = sys__dictionary__create();
    if (dictionary == 0ull) return 0ull;           /* the making already raised */

    uint64_t from = 0ull;
    for (uint64_t i = 0ull; i <= length; ++i) {
        /* The end of a line is a `\n` or the end of the text — an unterminated last line is still a
         * line, because refusing a file that does not end in a newline would be hostile and says
         * nothing about whether its contents are well formed. */
        if (i != length && bytes[i] != (uint8_t)'\n') continue;

        uint64_t to = i;
        if (to > from && bytes[to - 1ull] == (uint8_t)'\r') --to;   /* ⚖ *"`\r` ignored, `\n` splits"* */

        if (to == from) {
            /* ⛔ THE ONE EMPTY LINE THAT IS NOT A MISTAKE IS THE ONE AFTER THE FILE'S LAST `\n`, and it
             * is not a line at all — it is what a terminator leaves behind. Anything else empty is a
             * blank line in the middle of a config, which is refused until somebody asks for it. */
            if (i == length) break;
            sys__fault__raise(0ull, SYS__SETTINGS__FAULT_EMPTY);
            (void)sys__heap_object__release(dictionary);
            return 0ull;
        }
        if (!sys__settings__zzprivate_line(dictionary, bytes, from, to)) {
            (void)sys__heap_object__release(dictionary);
            return 0ull;
        }
        from = i + 1ull;
    }

    if (!sys__dictionary__seal(dictionary)) {
        (void)sys__heap_object__release(dictionary);
        return 0ull;
    }
    return dictionary;
}

/* What an init asks with: the string a key spells, BORROWED, or zero when the config did not name it.
 * ⛳ THE KEY IS TEXT AT THE CALL SITE AND A HASH HERE, which is the whole reason the parse keeps no key
 * text — a reader always knows the name it wants, so both sides compute the same 512 bits and the
 * dictionary compares eight words.
 *
 * ⭐⭐ BORROWED, AND THE SEAL IS WHAT LICENSES IT. `sys__dictionary__get` hands back a HOLD, which is
 * right for a table somebody may still write; `zzpackage_find` hands back the value and takes nothing,
 * and the file that publishes it says what that costs: it is *"for a dictionary nobody writes while the
 * answer is in use: a sealed one"*. The settings are sealed at the end of the parse and live in a
 * register row for the machine's lifetime, so there is no moment at which this answer can be displaced.
 * ⛔ USING `get` HERE WAS A LEAK AND THE HARNESS CAUGHT IT: every read took a hold nobody gave back, so
 * a value's count climbed once per question asked. An init that reads its sizing twice would have left
 * the string alive forever — invisible, because nothing about a settings value is ever released anyway.
 * ⇒ ★ A BORROWED READER IS NOT AN OPTIMISATION HERE, IT IS THE CORRECT ONE: the alternative obliges
 * every caller to release something it did not ask to own, and the first caller to forget is silent. */
static __device__ inline uint64_t sys__settings__value(uint64_t settings,
                                                        const uint8_t* name, uint64_t length) {
    sys__heap_node key;
    sys__string__hash(name, length, &key);
    bool found = false;
    const sys__heap_node value = sys__dictionary__zzpackage_find(settings, &key, &found);
    if (!found || value.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return value.args[0];
}

/* A value as a number, inside the engine — what an init reads a size with. Refuses anything that is not
 * decimal digits and anything that would wrap.
 * ⛳ ITS HOST TWIN IS `sys__settings__host_number` FORTY LINES DOWN, and the pair is covered by the same
 * differential test as the readers: a size the boot allocated from and an init checked against had
 * better be the same number, and a disagreement between these two is exactly how it would not be. */
static __device__ inline bool sys__settings__number(uint64_t string, uint64_t* answer) {
    if (answer == 0 || !sys__string__is(string)) return false;
    const uint8_t* bytes  = sys__string__bytes(string);
    const uint64_t length = sys__string__length(string);
    if (length == 0ull) return false;
    uint64_t made = 0ull;
    for (uint64_t i = 0ull; i < length; ++i) {
        if (bytes[i] < (uint8_t)'0' || bytes[i] > (uint8_t)'9') return false;
        const uint64_t digit = (uint64_t)(bytes[i] - (uint8_t)'0');
        if (made > (0xFFFFFFFFFFFFFFFFull - digit) / 10ull) return false;
        made = made * 10ull + digit;
    }
    *answer = made;
    return true;
}

/* The settings the boot left in the register, or zero when nothing stood them up.
 * ⛳ THE HOLD IS GIVEN BACK IMMEDIATELY, which is the same thing `sys__symbol` does with its three table
 * rows and rests on the same premise: the row holds the dictionary for the machine's life, so a reader
 * borrowing it cannot outlive what it borrowed. `REASONED`, and what would falsify it is a program that
 * overwrites the row — which is allowed, ⚖ deliberately, and would be the day this needs a real hold. */
/* ⭐ ONE DICTIONARY PER WORKER: the row names an array of them, and a reader is handed its own worker's — a
 * worker's settings are the sizes of ITS device's rooms. */
static __device__ inline uint64_t sys__settings__published(void) {
    const sys__heap_node held = sys__system_register__get(SYS__SYSTEM_REGISTER__SETTINGS);
    if (sys__heap_node__carries_reference(held.dtype)) (void)sys__heap_object__release(held.args[0]);
    if (held.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    if (sys__dictionary__is(held.args[0])) return held.args[0];       /* one dictionary: a machine of one worker */
    if (!sys__node_array__is(held.args[0])) return 0ull;
    const sys__heap_node mine = sys__node_array__borrow(held.args[0], (uint64_t)sys__silicon__worker());
    if (mine.dtype != SYS__KIND__OBJECT_REFERENCE || !sys__dictionary__is(mine.args[0])) return 0ull;
    return mine.args[0];
}

/* Does this key end in that suffix. The engine's half of a test the host also makes; both are generated
 * from `SYS__SETTINGS__UNIT_LIST`, so what they compare against cannot come apart. */
static __device__ inline bool sys__settings__zzprivate_tail(const uint8_t* key, uint64_t spelled,
                                                            const char* tail, uint64_t tail_length) {
    if (spelled < tail_length) return false;
    const uint64_t from = spelled - tail_length;
    for (uint64_t i = 0ull; i < tail_length; ++i)
        if (key[from + i] != (uint8_t)tail[i]) return false;
    return true;
}

/* The scale this key's own suffix spells, or false when it spells none. */
static __device__ inline bool sys__settings__zzprivate_scale(const uint8_t* key, uint64_t spelled,
                                                             unsigned* shift) {
#define SYS__SETTINGS__ZZPRIVATE_SCALE_ROW(tail, bits)                                                   \
    if (sys__settings__zzprivate_tail(key, spelled, tail, sizeof(tail) - 1ull)) {                        \
        *shift = (bits); return true;                                                                    \
    }
    SYS__SETTINGS__UNIT_LIST(SYS__SETTINGS__ZZPRIVATE_SCALE_ROW)
#undef SYS__SETTINGS__ZZPRIVATE_SCALE_ROW
    return false;
}

/* ══ WHAT AN INIT ACTUALLY ASKS — AND IT IS A PAIR, BECAUSE THE HOST'S IS ════════════════════════════
 *
 * ⛔⛔ THE CARD HAD ONE READER AND THE HOST HAD TWO, AND THAT WAS THE ASYMMETRY. `sys__settings__figure`
 * answered the RAW FIGURE: it was right for `__size_bytes`, whose shift is 0, and for `__qty`, which
 * names no unit — the only two keys anything inside the engine had ever asked for. Handed
 * `nn__expert_cache_l1__size_mb` it would have answered MEBIBYTES TO A CALLER EXPECTING BYTES, and the
 * arena would have been carved at one 1,048,576th of the room the host took for it, with the span's
 * total still right, every buffer run still right and every address still resolving.
 * ⇒ ★★ A DUPLICATION HELD BY A DIFFERENTIAL TEST IS HELD ONLY ON WHAT THE TEST COMPARES. That test
 * required the same PAIRS out of both readers — a pair being a key's text and a value's text — and a
 * unit is neither. The one thing the two tiers could disagree about was the one thing the check that
 * exists to stop them disagreeing did not look at. ▶ it now requires the same BYTES.
 * ⇒ ★ AND THE TELL WAS IN THIS FILE'S OWN PROSE: `host_count` names `nn__expert_cache_l1__size_mb` as
 * its worked example of the mistake with "no later symptom that names its cause" — beside a tier that
 * could not have refused it.
 *
 * ⛳ THE TWO REFUSE EACH OTHER'S KEYS, exactly as the host pair does, and for the reason the host pair
 * gives: a key naming a unit is a SIZE and a key naming none is a COUNT, so a caller reaching for the
 * wrong door is told rather than handed a layer count multiplied by 1<<20.
 * ⛔ ONE THING THE ENGINE'S PAIR CANNOT DO THAT THE HOST'S CAN: say WHICH mistake. The host answers
 * ABSENT · FOUND · MALFORMED; these answer a bool, because that is the contract every engine-side reader
 * here has and an init's response to all three is its own fault code either way. ⛳ The key is still
 * judged BEFORE the dictionary is asked, so a caller's bad key is bad whatever the config says and
 * cannot be masked by a file that happens not to name it. ══════════════════════════════════════════ */

/* A SIZE, answered in BYTES, in whatever unit its own key spells. */
static __device__ inline bool sys__settings__size(const uint8_t* name, uint64_t length,
                                                  uint64_t* bytes) {
    if (bytes == 0 || name == 0) return false;
    unsigned shift = 0u;
    if (!sys__settings__zzprivate_scale(name, length, &shift)) return false;  /* a size, of a unitless key */
    const uint64_t settings = sys__settings__published();
    if (settings == 0ull) return false;
    const uint64_t value = sys__settings__value(settings, name, length);
    if (value == 0ull) return false;
    uint64_t figure = 0ull;
    if (!sys__settings__number(value, &figure) || figure == 0ull) return false;
    if (figure > (0xFFFFFFFFFFFFFFFFull >> shift)) return false;              /* would wrap on the shift */
    *bytes = figure << shift;
    return true;
}

/* A COUNT — what a key with NO unit on it spells. */
static __device__ inline bool sys__settings__count(const uint8_t* name, uint64_t length,
                                                   uint64_t* answer) {
    if (answer == 0 || name == 0) return false;
    unsigned shift = 0u;
    if (sys__settings__zzprivate_scale(name, length, &shift)) return false;   /* that key names a size */
    const uint64_t settings = sys__settings__published();
    if (settings == 0ull) return false;
    const uint64_t value = sys__settings__value(settings, name, length);
    if (value == 0ull) return false;
    return sys__settings__number(value, answer);
}

/* ══ THE HOST'S HALF — THE SAME FORMAT, READ WHERE THERE ARE NO REGISTERS ════════════════════════════
 *
 * ⚖ ARCHITECT: *"i do wonder if each abi needs to read the json on its own, i think so because
 * it is a host method so we dont care about register cost. so yea each init method has a reference to the
 * file and can set up their own area."*
 *
 * ⛔⛔ SO THERE ARE TWO READERS OF ONE FORMAT, AND THAT IS A RULING RATHER THAN AN ACCIDENT. What the
 * host reads is WHATEVER IT NEEDS: `boot_commands` always, because the engine never sees that section at
 * all, and `device_params` whenever a figure is acted on by both tiers. ⭐ THE SPLIT STANDS ON
 * OWNERSHIP: a host reads what it needs and the engine is told only the figures it acts on.
 * ⛳ BOTH READERS RUN ON THE CPU. "The host" in this file is the one that runs before there is an engine,
 * over plain bytes; the other runs inside the engine, over its dictionary, when a package's init asks.
 * ⛳ THE SECOND CASE IS NOT A LOOPHOLE — it is the sentence at the foot of this block,
 * and `nn`'s span is its first tenant: the host SIZES an allocation before there is an engine and the
 * opcode init CARVES the same allocation after there is one, off one set of keys nobody writes twice.
 * ⛳⛳ AND THEY SIT IN ONE FILE ON PURPOSE. A duplication nobody can see both halves of is a duplication
 * that drifts; these are forty lines apart, so a change to one is read next to the other.
 * ⛔ THAT IS STILL NOT ENOUGH, AND THE TREE HAS SAID SO BEFORE: adjacency is a convenience for whoever
 * happens to look. THE CHECK IS THE DIFFERENTIAL TEST — one text through both readers, the same pairs
 * required out — because what stops a blessed duplication from drifting is something that FAILS when it
 * does, never discipline. ▶ `test_settings_both_readers` in the host harness.
 *
 * ── WHAT THE HOST READS, AND WHAT IT REFUSES TO UNDERSTAND ──────────────────────────────────────────
 * The file is two sections of the lines above:
 *     boot_commands{ … }    what needs NO standing engine — read HERE
 *     device_params{ … }    what a package's init reads inside the engine — handed over BYTES VERBATIM
 * ⭐ THE HOST NEVER RENDERS THE DEVICE'S SECTION, IT SLICES IT. `sys__settings__from_string` receives
 * exactly the text a human wrote, so there is no serialiser to disagree with a parser — the one shape of
 * drift this design cannot have is the one that would be hardest to see.
 * ⛳ It may still READ a `device_param` — the host holds the whole file either way, and the split is about
 * who consumes a value at RUNTIME, not about who may look at it. That is what lets a package's room size
 * live in exactly one place while both sides use it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The bytes of one named section, without its header or its closing brace. A section opens with
 * `<name>{` and closes with `}`, each at the start of its own line — the whole of the structure, because
 * sections do not nest and a value cannot contain a newline.
 * ⛳ THE STRUCTURE IS THE HOST'S AND THE PAIRS ARE THE FORMAT'S, which is the seam that keeps this from
 * being a second implementation of anything: finding a brace is not reading a setting. */
static inline int sys__settings__host_section(const char* text, uint64_t text_length, const char* name,
                                              const char** at, uint64_t* length) {
    if (text == 0 || name == 0 || at == 0 || length == 0) return SYS__SETTINGS__READ_ABSENT;
    uint64_t spelled = 0ull;
    while (name[spelled] != '\0') ++spelled;
    uint64_t i = 0ull;
    while (i < text_length) {
        /* the start of a line */
        uint64_t end = i;
        while (end < text_length && text[end] != '\n') ++end;
        uint64_t stop = end;
        if (stop > i && text[stop - 1ull] == '\r') --stop;
        if (stop - i == spelled + 1ull && text[stop - 1ull] == '{') {
            bool same = true;
            for (uint64_t k = 0ull; k < spelled; ++k) if (text[i + k] != name[k]) { same = false; break; }
            if (same) {
                /* the body runs to the line that is a bare `}` */
                const uint64_t from = (end < text_length) ? end + 1ull : text_length;
                uint64_t j = from;
                while (j < text_length) {
                    uint64_t line_end = j;
                    while (line_end < text_length && text[line_end] != '\n') ++line_end;
                    uint64_t line_stop = line_end;
                    if (line_stop > j && text[line_stop - 1ull] == '\r') --line_stop;
                    if (line_stop - j == 1ull && text[j] == '}') {
                        *at = text + from; *length = j - from;
                        return SYS__SETTINGS__READ_FOUND;
                    }
                    j = (line_end < text_length) ? line_end + 1ull : text_length;
                }
                return SYS__SETTINGS__READ_MALFORMED;   /* opened and never closed */
            }
        }
        i = (end < text_length) ? end + 1ull : text_length;
    }
    return SYS__SETTINGS__READ_ABSENT;
}

/* Walk a section's pairs. `cursor` starts at zero and is the caller's; the key and value come back as
 * spans into the section, never copied.
 * ⛔ IT APPLIES THE CARD'S RULES, NOT A RELAXED VERSION OF THEM — a blank line, a line with no colon, an
 * empty key or value, and any byte outside `0x21..0x7E` are all `REFUSED`, at the same byte and for the
 * same reason. A host reader that were more forgiving would let a file through here and fail in the engine,
 * which is the one failure mode a second reader must not add. ⛳ The duplicate-key refusal is the
 * dictionary's and cannot be made here without a table; the differential test is what covers it. */
static inline int sys__settings__host_next(const char* at, uint64_t length, uint64_t* cursor,
                                           const char** key, uint64_t* key_length,
                                           const char** value, uint64_t* value_length) {
    if (at == 0 || cursor == 0 || key == 0 || key_length == 0 || value == 0 || value_length == 0) {
        return SYS__SETTINGS__HOST_REFUSED;
    }
    if (*cursor >= length) return SYS__SETTINGS__HOST_DONE;

    const uint64_t from = *cursor;
    uint64_t end = from;
    while (end < length && at[end] != '\n') ++end;
    uint64_t stop = end;
    if (stop > from && at[stop - 1ull] == '\r') --stop;
    *cursor = (end < length) ? end + 1ull : length;

    if (stop == from) {
        /* The empty line after the section's last `\n` is what a terminator leaves behind, not a line. */
        if (*cursor >= length) return SYS__SETTINGS__HOST_DONE;
        return SYS__SETTINGS__HOST_REFUSED;
    }
    uint64_t colon = stop;
    for (uint64_t i = from; i < stop; ++i) {
        const unsigned char byte = (unsigned char)at[i];
        if (byte < 0x21u || byte > 0x7Eu) return SYS__SETTINGS__HOST_REFUSED;
        if (at[i] == ':' && colon == stop) colon = i;
    }
    if (colon == stop || colon == from || colon + 1ull == stop) return SYS__SETTINGS__HOST_REFUSED;

    *key = at + from;            *key_length   = colon - from;
    *value = at + colon + 1ull;  *value_length = stop - colon - 1ull;
    return SYS__SETTINGS__HOST_PAIR;
}

/* One key's value out of a section — and THREE answers, for the reason the section finder has three.
 * ⛔⛔ THIS RETURNED A PLAIN BOOL UNTIL A DEVICE TEST CAUGHT IT, AND THE FAILURE WAS THE SAME SHAPE AS THE
 * TRUNCATED SECTION ONE FLOOR UP. A line that will not parse ended the walk and answered `false`, which
 * every caller read as "the key is not there" — so `boot_commands{ nn__alloc_size_mb }`, a key with its
 * colon left off, BOOTED AT THE DEFAULT SIZE. ⇒ ★ "NOT FOUND" AND "UNREADABLE" ARE THE TWO ANSWERS A
 * CONFIG READER MUST NEVER CONFLATE, and this file conflated them twice in two days through two different
 * functions — which is what says the distinction belongs in the TYPE rather than in each caller's care.
 * ⛳ AND THE WALK STOPS AT THE FIRST BAD LINE rather than reading past it: a section with a typo in it is
 * not a section some of whose keys are trustworthy. */
static inline int sys__settings__host_value(const char* at, uint64_t length, const char* key,
                                            const char** value, uint64_t* value_length) {
    if (key == 0) return SYS__SETTINGS__READ_MALFORMED;
    uint64_t spelled = 0ull;
    while (key[spelled] != '\0') ++spelled;
    uint64_t cursor = 0ull;
    const char* k = 0; uint64_t kl = 0ull;
    for (;;) {
        const int step = sys__settings__host_next(at, length, &cursor, &k, &kl, value, value_length);
        if (step == SYS__SETTINGS__HOST_DONE)    return SYS__SETTINGS__READ_ABSENT;
        if (step == SYS__SETTINGS__HOST_REFUSED) return SYS__SETTINGS__READ_MALFORMED;
        if (kl != spelled) continue;
        bool same = true;
        for (uint64_t i = 0ull; i < spelled; ++i) if (k[i] != key[i]) { same = false; break; }
        if (same) return SYS__SETTINGS__READ_FOUND;
    }
}

/* A value as a number, which is what every figure in `boot_commands` is. Refuses anything that is not
 * digits, and anything that would not fit — a size read wrong is an allocation made wrong.
 * ⛳ DECIMAL ONLY, because a config is written by a person and `0x` in one is more likely a mistake than
 * an intention; the engine's own reader will want the same rule when an init reads a device_param. */
static inline bool sys__settings__host_number(const char* value, uint64_t value_length, uint64_t* answer) {
    if (value == 0 || answer == 0 || value_length == 0ull) return false;
    uint64_t made = 0ull;
    for (uint64_t i = 0ull; i < value_length; ++i) {
        if (value[i] < '0' || value[i] > '9') return false;
        const uint64_t digit = (uint64_t)(value[i] - '0');
        if (made > (0xFFFFFFFFFFFFFFFFull - digit) / 10ull) return false;   /* would wrap */
        made = made * 10ull + digit;
    }
    *answer = made;
    return true;
}

/* Does this key end in that suffix. Spelled out rather than reached for, because there is no string
 * library on the far side of this file and one comparison loop is cheaper than owning a dependency. */
static inline bool sys__settings__zzprivate_host_tail(const char* key, uint64_t spelled,
                                                      const char* tail, uint64_t tail_length) {
    if (spelled < tail_length) return false;
    const uint64_t from = spelled - tail_length;
    for (uint64_t i = 0ull; i < tail_length; ++i) if (key[from + i] != tail[i]) return false;
    return true;
}

/* ⭐⭐ THE UNIT COMES OFF THE KEY, NOT FROM AN ARGUMENT BESIDE IT. ⚖ Agreed: every size key in
 * this format carries its own unit — `_bytes` · `_kb` · `_mb` — and the reader takes its scale from the
 * key it was handed.
 * ⇒ ★ A UNIT PASSED SEPARATELY FROM ITS KEY IS A UNIT THAT CAN BE PASSED WRONG. Here the pairing cannot
 * come apart: the same characters that name the key in the config name it in the call, and the scale is
 * read out of those characters rather than remembered alongside them. There is nothing to keep in step.
 * ⛔ A KEY ENDING IN NONE OF THEM IS REFUSED rather than assumed to be bytes. Asking a SIZE of
 * `nn__buffer_main__qty` is a mistake no default can improve — and refusing is the direction that can be
 * relaxed later without changing what an already-working file means.
 * ⛳ THE SUFFIXES THEMSELVES ARE NOT WRITTEN HERE — this expands `SYS__SETTINGS__UNIT_LIST`, which the
 * engine's `zzprivate_scale` expands too. That is what stops the one asymmetry this file actually had. */
static inline bool sys__settings__zzprivate_host_scale(const char* key, uint64_t spelled,
                                                       unsigned* shift) {
#define SYS__SETTINGS__ZZPRIVATE_HOST_SCALE_ROW(tail, bits)                                              \
    if (sys__settings__zzprivate_host_tail(key, spelled, tail, sizeof(tail) - 1ull)) {                   \
        *shift = (bits); return true;                                                                    \
    }
    SYS__SETTINGS__UNIT_LIST(SYS__SETTINGS__ZZPRIVATE_HOST_SCALE_ROW)
#undef SYS__SETTINGS__ZZPRIVATE_HOST_SCALE_ROW
    return false;
}

/* ⭐ WHAT BOTH SIDES OF THE BOOT ACTUALLY ASK: a size, by name, out of a named section, in whatever unit
 * its own key spells, and answered in BYTES. ⚖ The keys are the architect's convention —
 * `<package>__alloc_size_mb`, `nn__buffer_main__size_bytes`, `nn__expert__size_q4_kb` — and they live in
 * `boot_commands` because an allocation needs no standing engine.
 * ⛔ THREE ANSWERS, AND THE CALLER REFUSES ON THE THIRD. `ABSENT` means nobody asked, which is how a
 * package keeps its default; `MALFORMED` covers a section cut short, a value that is not a figure, AND a
 * key with no unit on it, because all three mean somebody said something that could not be read — and
 * quietly using the default THERE is how a machine boots at a size nobody chose.
 * ⛳ THE KEY IS JUDGED BEFORE THE FILE IS OPENED, which makes the answer deterministic: a caller's bad key
 * is bad whatever the config says, so it cannot be masked by a file that happens not to name it. */
static inline int sys__settings__host_size(const char* config, uint64_t length,
                                           const char* section_name, const char* key,
                                           uint64_t* bytes) {
    if (bytes == 0 || key == 0) return SYS__SETTINGS__READ_MALFORMED;
    uint64_t spelled = 0ull;
    while (key[spelled] != '\0') ++spelled;
    unsigned shift = 0u;
    if (!sys__settings__zzprivate_host_scale(key, spelled, &shift))
        return SYS__SETTINGS__READ_MALFORMED;               /* a size asked of a key that names no unit */
    if (config == 0 || length == 0ull) return SYS__SETTINGS__READ_ABSENT;
    const char* section = 0; uint64_t section_length = 0ull;
    const int found = sys__settings__host_section(config, length, section_name, &section, &section_length);
    if (found != SYS__SETTINGS__READ_FOUND) return found;   /* ABSENT, or TRUNCATED and refused above */
    const char* value = 0; uint64_t value_length = 0ull;
    const int said = sys__settings__host_value(section, section_length, key, &value, &value_length);
    if (said != SYS__SETTINGS__READ_FOUND) return said;   /* ABSENT, or MALFORMED and refused by the caller */
    uint64_t figure = 0ull;
    if (!sys__settings__host_number(value, value_length, &figure) || figure == 0ull)
        return SYS__SETTINGS__READ_MALFORMED;               /* said something unreadable */
    if (figure > (0xFFFFFFFFFFFFFFFFull >> shift)) return SYS__SETTINGS__READ_MALFORMED;   /* would wrap */
    *bytes = figure << shift;
    return SYS__SETTINGS__READ_FOUND;
}

/* ⭐ A PLAIN FIGURE OUT OF A NAMED SECTION — a COUNT, which is what a key with NO unit on it spells.
 * ⛳⛳ THE MIRROR OF `host_size`, AND THE MIRRORING IS THE POINT RATHER THAN A TIDINESS: a key naming a
 * unit is a size and a key naming none is a count, so between them these two functions refuse each other's
 * keys. Asking `host_count` for `nn__expert_cache_l1__size_mb` is refused, and asking `host_size` for
 * `nn__model__kv_cache_layers_fullattn` is refused — in both directions, by the same one test.
 * ⇒ ★ A CALLER THAT REACHES FOR THE WRONG READER IS TOLD SO, where the cost of not being told is a layer
 * count multiplied by 1<<20 or an arena sized in bytes somebody wrote as mebibytes. Neither has a later
 * symptom that names its cause. */
static inline int sys__settings__host_count(const char* config, uint64_t length,
                                            const char* section_name, const char* key,
                                            uint64_t* answer) {
    if (answer == 0 || key == 0) return SYS__SETTINGS__READ_MALFORMED;
    uint64_t spelled = 0ull;
    while (key[spelled] != '\0') ++spelled;
    unsigned shift = 0u;
    if (sys__settings__zzprivate_host_scale(key, spelled, &shift))
        return SYS__SETTINGS__READ_MALFORMED;               /* a key naming a unit is a size, not a count */
    if (config == 0 || length == 0ull) return SYS__SETTINGS__READ_ABSENT;
    const char* section = 0; uint64_t section_length = 0ull;
    const int found = sys__settings__host_section(config, length, section_name, &section, &section_length);
    if (found != SYS__SETTINGS__READ_FOUND) return found;
    const char* value = 0; uint64_t value_length = 0ull;
    const int said = sys__settings__host_value(section, section_length, key, &value, &value_length);
    if (said != SYS__SETTINGS__READ_FOUND) return said;
    if (!sys__settings__host_number(value, value_length, answer)) return SYS__SETTINGS__READ_MALFORMED;
    return SYS__SETTINGS__READ_FOUND;
}

/* Does this key begin with that prefix — the head to `zzprivate_host_tail`'s tail. */
static inline bool sys__settings__zzprivate_host_head(const char* key, uint64_t spelled,
                                                      const char* head, uint64_t head_length) {
    if (spelled < head_length) return false;
    for (uint64_t i = 0ull; i < head_length; ++i) if (key[i] != head[i]) return false;
    return true;
}

/* ⭐⭐ EVERY KEY UNDER A PREFIX, ADDED UP — AND THE REASON IT IS A SUM RATHER THAN A TOTAL SOMEBODY WRITES.
 * ⚖ ARCHITECT, on the layer counts: *"fullattn + deltanet and the sum is the layer total, if
 * we will add a new kv cache mechanism it will be another layer option."*
 * ⇒ ⭐ A STATED TOTAL AND ITS PARTS ARE TWO WRITABLE NUMBERS OBLIGED TO AGREE, WITH NOTHING CHECKING THEM,
 * which is the rule this tree has now applied three times — and an interval (`every 4th layer`) would need
 * its derivation rewritten the day a third mechanism arrives. A sum over a prefix needs neither: a new
 * mechanism is a new key in the file and NO CODE HERE AT ALL.
 * ⛳ WHICH IS WHY IT WALKS PAIRS RATHER THAN LOOKING KEYS UP: `host_next` enumerates, so this never has to
 * know what the mechanisms are called. The same property the package roster has, for the same reason.
 * ⛔ NO MATCH IS `ABSENT` AND NOT A ZERO, because those are different facts — a config that names no layer
 * mechanism has said nothing, while `..._fullattn:0` is a real model with no full-attention layers. A
 * zero VALUE therefore sums normally; it is only zero MATCHES that answers absent.
 * ⛔ AND A BAD LINE ANYWHERE IN THE SECTION REFUSES THE WHOLE SUM, prefix or not — the same ruling
 * `host_value` runs under: a section with a typo in it is not a section some of whose keys are
 * trustworthy, and a sum is exactly where a quietly-skipped line would go unnoticed. */
static inline int sys__settings__host_prefix_sum(const char* config, uint64_t length,
                                                 const char* section_name, const char* prefix,
                                                 uint64_t* answer) {
    if (answer == 0 || prefix == 0) return SYS__SETTINGS__READ_MALFORMED;
    if (config == 0 || length == 0ull) return SYS__SETTINGS__READ_ABSENT;
    uint64_t spelled = 0ull;
    while (prefix[spelled] != '\0') ++spelled;
    if (spelled == 0ull) return SYS__SETTINGS__READ_MALFORMED;   /* everything is not a prefix */

    const char* section = 0; uint64_t section_length = 0ull;
    const int found = sys__settings__host_section(config, length, section_name, &section, &section_length);
    if (found != SYS__SETTINGS__READ_FOUND) return found;

    uint64_t cursor = 0ull, made = 0ull, matched = 0ull;
    const char *k = 0, *v = 0;
    uint64_t kl = 0ull, vl = 0ull;
    for (;;) {
        const int step = sys__settings__host_next(section, section_length, &cursor, &k, &kl, &v, &vl);
        if (step == SYS__SETTINGS__HOST_DONE)    break;
        if (step == SYS__SETTINGS__HOST_REFUSED) return SYS__SETTINGS__READ_MALFORMED;
        if (!sys__settings__zzprivate_host_head(k, kl, prefix, spelled)) continue;
        uint64_t figure = 0ull;
        if (!sys__settings__host_number(v, vl, &figure)) return SYS__SETTINGS__READ_MALFORMED;
        if (made > 0xFFFFFFFFFFFFFFFFull - figure) return SYS__SETTINGS__READ_MALFORMED;   /* would wrap */
        made += figure;
        matched += 1ull;
    }
    if (matched == 0ull) return SYS__SETTINGS__READ_ABSENT;
    *answer = made;
    return SYS__SETTINGS__READ_FOUND;
}

/* The value of one key spelled in three parts — `<prefix><middle><tail>` — without assembling it. Every
 * lookup below names a key that is not a string anywhere: the middle is a SPAN out of another key, so
 * building the name would need a scratch buffer, which is the one thing this file has refused throughout.
 * Comparing the three parts in place costs the same and owns nothing. */
static inline int sys__settings__zzprivate_host_split_value(const char* section, uint64_t section_length,
                                                            const char* prefix, uint64_t prefix_length,
                                                            const char* middle, uint64_t middle_length,
                                                            const char* tail, uint64_t tail_length,
                                                            const char** value, uint64_t* value_length) {
    uint64_t cursor = 0ull;
    const char *k = 0;
    uint64_t kl = 0ull;
    for (;;) {
        const int step = sys__settings__host_next(section, section_length, &cursor,
                                                  &k, &kl, value, value_length);
        if (step == SYS__SETTINGS__HOST_DONE)    return SYS__SETTINGS__READ_ABSENT;
        if (step == SYS__SETTINGS__HOST_REFUSED) return SYS__SETTINGS__READ_MALFORMED;
        if (kl != prefix_length + middle_length + tail_length)                       continue;
        if (!sys__settings__zzprivate_host_head(k, kl, prefix, prefix_length))       continue;
        if (!sys__settings__zzprivate_host_tail(k, kl, tail, tail_length))           continue;
        bool same = true;
        for (uint64_t i = 0ull; i < middle_length; ++i)
            if (k[prefix_length + i] != middle[i]) { same = false; break; }
        if (same) return SYS__SETTINGS__READ_FOUND;
    }
}

/* ⭐⭐ A SUM OF PRODUCTS OVER WHATEVER CLASSES THE FILE NAMES — `Σ <prefix><class><tail_a> ×
 * <prefix><class><tail_b>`. What the buffer classes need: a size and a count each, and the room they want
 * is the sum of one run per class.
 * ⚖ ARCHITECT, on the class set: *"main/resid/alt were the one of v1, if we need to add other
 * two it is fine by me."* ⇒ ⭐ SO ADDING ONE IS TWO LINES IN A CONFIG AND NO CODE HERE, which is the same
 * property §①ⓐ's layer sum has and was chosen for. A fixed list of three class NAMES in this file would
 * have made the architect's "fine by me" cost a rebuild.
 * ⛔ A HALF-DECLARED CLASS IS MALFORMED, IN BOTH DIRECTIONS — a size with no count, and a count with no
 * size. That is the whole reason the counts are compared at the end rather than the products merely being
 * summed: an unpaired `__qty` is invisible to a walk that only iterates sizes, and it means somebody
 * renamed one of the two lines. ⛳ Silently ignoring it would take a span one class short of what the file
 * asks for, which is the shape of mistake that surfaces as a refusal three layers away.
 * ⛔ THE TWO TAILS MUST NOT BE SUFFIXES OF ONE ANOTHER, or a key would match both and be its own partner.
 * `__size_bytes` and `__qty` are not, and a pair that were would be a caller's error, not a file's. */
static inline int sys__settings__host_paired_sum(const char* config, uint64_t length,
                                                 const char* section_name, const char* prefix,
                                                 const char* tail_a, const char* tail_b,
                                                 uint64_t* answer) {
    if (answer == 0 || prefix == 0 || tail_a == 0 || tail_b == 0) return SYS__SETTINGS__READ_MALFORMED;
    if (config == 0 || length == 0ull) return SYS__SETTINGS__READ_ABSENT;
    uint64_t spelled = 0ull, spelled_a = 0ull, spelled_b = 0ull;
    while (prefix[spelled]   != '\0') ++spelled;
    while (tail_a[spelled_a] != '\0') ++spelled_a;
    while (tail_b[spelled_b] != '\0') ++spelled_b;
    if (spelled == 0ull || spelled_a == 0ull || spelled_b == 0ull) return SYS__SETTINGS__READ_MALFORMED;

    const char* section = 0; uint64_t section_length = 0ull;
    const int found = sys__settings__host_section(config, length, section_name, &section, &section_length);
    if (found != SYS__SETTINGS__READ_FOUND) return found;

    uint64_t cursor = 0ull, made = 0ull, sides_a = 0ull, sides_b = 0ull;
    const char *k = 0, *v = 0;
    uint64_t kl = 0ull, vl = 0ull;
    for (;;) {
        const int step = sys__settings__host_next(section, section_length, &cursor, &k, &kl, &v, &vl);
        if (step == SYS__SETTINGS__HOST_DONE)    break;
        if (step == SYS__SETTINGS__HOST_REFUSED) return SYS__SETTINGS__READ_MALFORMED;
        if (!sys__settings__zzprivate_host_head(k, kl, prefix, spelled)) continue;
        if (sys__settings__zzprivate_host_tail(k, kl, tail_b, spelled_b)) sides_b += 1ull;
        if (!sys__settings__zzprivate_host_tail(k, kl, tail_a, spelled_a)) continue;
        if (kl <= spelled + spelled_a) return SYS__SETTINGS__READ_MALFORMED;   /* a class with no name */
        sides_a += 1ull;

        const char* middle = k + spelled;
        const uint64_t middle_length = kl - spelled - spelled_a;
        const char* partner = 0; uint64_t partner_length = 0ull;
        const int said = sys__settings__zzprivate_host_split_value(section, section_length,
                                                                   prefix, spelled,
                                                                   middle, middle_length,
                                                                   tail_b, spelled_b,
                                                                   &partner, &partner_length);
        if (said != SYS__SETTINGS__READ_FOUND) return SYS__SETTINGS__READ_MALFORMED;   /* half a class */

        uint64_t a = 0ull, b = 0ull;
        if (!sys__settings__host_number(v, vl, &a))                     return SYS__SETTINGS__READ_MALFORMED;
        if (!sys__settings__host_number(partner, partner_length, &b))   return SYS__SETTINGS__READ_MALFORMED;
        if (a != 0ull && b > 0xFFFFFFFFFFFFFFFFull / a) return SYS__SETTINGS__READ_MALFORMED;   /* would wrap */
        const uint64_t run = a * b;
        if (made > 0xFFFFFFFFFFFFFFFFull - run) return SYS__SETTINGS__READ_MALFORMED;           /* would wrap */
        made += run;
    }
    if (sides_a == 0ull && sides_b == 0ull) return SYS__SETTINGS__READ_ABSENT;
    if (sides_a != sides_b) return SYS__SETTINGS__READ_MALFORMED;   /* a count naming a class with no size */
    *answer = made;
    return SYS__SETTINGS__READ_FOUND;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SETTINGS_CUH */
