#ifndef SILVANN__PACKAGES_NN_CPU_BUFFER__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_BUFFER__HEADER_CUH
#include "../contracts/objects/buffer.cuh" /* its constants and fault words */
/* ══ nn — WHAT A BUFFER CLASS IS, AND WHERE ITS ID COMES FROM ════════════════════════════════════════
 *
 * ⚖⚖ THE INDEXING, RULED: *"ideally for the buffer setup i would like to have a two tier array
 * so we have `buffer[buffer_kind][buffer_num]`, and adding a buffer kind is an enum addition plus one
 * config line plus the init code to instantiate it from the config."*
 *
 * ⭐⭐ THIS FILE IS THE FIRST HALF OF THAT — THE `[buffer_kind]` AXIS, AND NOTHING ELSE. One row per class
 * is the only place a class exists: from it come the enum, the count, and the two config keys it is sized
 * by. The second half is the run of bytes, which is arithmetic and not a structure, and it is carved from
 * `nn`'s span by the opcode init. ▶ `activation_buffer_pool.md` §⓪ⓐ.
 *
 * ── ⛔⛔ WHY THE IDS CANNOT COME FROM THE CONFIG FILE, WHICH IS THE THING THIS FILE EXISTS TO FIX ──────
 * `sys__settings__host_paired_sum` sums `size x qty` over whatever classes a file NAMES. It answers a
 * TOTAL and it never says *"resid is 0"* — and a file's ORDER is not an identity: reorder the lines and
 * every class renumbers, silently, while the total stays exactly right.
 * ⇒ ★★ A SUM OVER A SET AND AN INDEX INTO IT ARE DIFFERENT QUESTIONS. Nothing about that sum was wrong;
 * it was never going to answer the second one. An id must be stable across runs and across files, so it
 * belongs where the build can see it, which is here.
 *
 * ── ⭐ THE ROW CARRIES THE NAME TWICE, AND EVERY WAY TO GET THAT WRONG IS CAUGHT ─────────────────────
 * The second column is the spelling the CONFIG uses and the third is the constant the CODE writes, which
 * is the same shape as the KINDS row next door — `X(PKG, 0, "nn__buffer", NN__KIND__BUFFER)` names its
 * kind twice for the same reason. Two spellings of one name would normally be this tree's least favourite
 * thing, so it is worth writing down that none of the three ways to break them can pass quietly:
 *     a constant misspelled     nothing names it and the use site fails to compile
 *     a constant duplicated     a repeated enumerator, which is a compile error at this row
 *     the SPELLING misspelled   the list asks the config for `nn__buffer_logtis__*`, the config spells
 *                               `logits`, the two totals disagree and the boot REFUSES — §⓪ⓑ's check,
 *                               arriving at the one moment there is something to be done about it
 * ⇒ ★ THE REDUNDANCY IS AFFORDABLE BECAUSE IT IS COVERED, not because it is small. The one thing no
 * compiler could see — a name that is a valid word and the wrong word — is exactly what the completeness
 * check was built for, and it was built for the other direction first.
 *
 * ── ⛔ AND THE IDS MUST BE DENSE AND IN LIST ORDER, WHICH IS GENERATED AND NOT ASKED FOR ──────────────
 * The class table is indexed BY THE ID and the runs are laid out as a running total IN LIST ORDER, so an
 * id that is not its own position silently hands every class another class's bytes. Nothing downstream
 * could tell: the total is still right, the array is still the right length, and every address resolves.
 * ⛳ SO THE CHECK IS GENERATED FROM THE LIST rather than trusted — the same move `PACKAGE_LIST` makes with
 * `PACKAGE_<name>_PRESENT`. ⇒ ★ WHEN THE LANGUAGE WILL NOT GENERATE A THING, GENERATE THE CHECK THAT IT
 * WAS DONE.
 *
 * ⛳ A SIXTH CLASS IS A ROW HERE AND TWO LINES IN A CONFIG. That is one step cheaper than the ruling asked
 * for: it asked for an enum addition, a config line and the init code, and the init code is generated from
 * the row. ⛔ WHAT IT IS NOT IS A CONFIG-ONLY CHANGE, and the reason is the whole subject of this file: a
 * class that is INDEXED needs an id, an id has to be stable across runs and across files, and only the
 * build can hand one out. The row is the price of being indexable, and it is one line.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* The two keys a class is sized by, spelled by STRINGIFICATION so neither can be misspelled in one place
 * and not the other, and so no key is ever assembled at runtime into a buffer this package would have to
 * own. ⛳ THE SUFFIXES ARE WHAT PICKS THE READER: `__size_bytes` names a unit and is a `host_size`, `__qty`
 * names none and is a `host_count`, and the two readers refuse each other's keys. */
#define NN__BUFFER__KEY_SIZE(name)  "nn__buffer_" #name "__size_bytes"
#define NN__BUFFER__KEY_QTY(name)   "nn__buffer_" #name "__qty"
#define NN__BUFFER__KEY_RAM_SIZE(name)  "nn__ram_buffer_" #name "__size_bytes"
#define NN__BUFFER__KEY_RAM_QTY(name)   "nn__ram_buffer_" #name "__qty"

/* This class's free list, or zero. ⛳ `hatch[nn][BUFFERS][kind]` — the whole of the lookup the ruling
 * asked for, and every hop is an index because every level is static. */
static __device__ inline uint64_t nn__buffer__zzpackage_freelist(uint64_t kind);

/* How many bytes one buffer of this class is, as the heap int the sizeof array holds. ⚖ *"a node array
 * where each node is an int ... so you can do buffer_sizeof[buffer_enum_id]"*. */
static __device__ inline uint64_t nn__buffer__sizeof(uint64_t kind);

/* Cut the span into classes and fill their lists. Runs in the OPCODE init, because every line of it makes
 * a heap object and those need a standing engine.
 * ⭐ `ended` ANSWERS THE FIRST BYTE AFTER THE LAST RUN, which is where the expert L1 arena begins. It is
 * the running total this walk already keeps, handed out rather than dropped — the alternative being a
 * second walk summing the same five products, i.e. a second copy of a number. ▶ `expert__header.cuh`.
 * ⛳ IT IS WRITTEN ON SUCCESS ONLY; a refused carve has no end to report and its caller reads nothing. */
static __device__ inline bool nn__buffer__zzpackage_carve(uint64_t span_at, uint64_t span_bytes,
                                                          uint64_t ram_at, uint64_t ram_card_at,
                                                          uint64_t ram_bytes, uint64_t* ended);

/* Where a buffer goes when nobody wants it: back in its own class's list. */
static __device__ __noinline__ void nn__buffer__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack);

/* `(nn__buffer__getnew  size)` — the smallest class that fits, or an error. */
static __device__ __noinline__ void nn__buffer__zzpackage_apply_getnew(sys__heap_node* base, uint64_t form);
/* `(nn__buffer__getnew_ram size)` — the smallest RAM class that fits: bytes the card writes and the program
 * reads where they are. ▶ `NN__BUFFER_RAM_CLASS_LIST`. */
static __device__ __noinline__ void nn__buffer__zzpackage_apply_getnew_ram(sys__heap_node* base, uint64_t form);
/* `(nn__buffer__read b)` on the DEVICE evaluator — where no verb writes a stamped value, so there is never
 * one to read. ▶ `cpu/opcodes/result__abi.cuh` for the verb itself. */

#endif /* SILVANN__PACKAGES_NN_CPU_BUFFER__HEADER_CUH */
