#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SETTINGS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SETTINGS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */

/* ⛳ GROUPED AND PREFIXED LIKE EVERY OTHER COMPONENT'S — `STG`, which no other word in the tree begins
 * with, so a log line says which component refused before anybody opens a file. */
#define SYS__SETTINGS__FAULT_KIND   0x5354474Eull   /* "STGN" — handed something that is Not a string  */
#define SYS__SETTINGS__FAULT_LINE   0x5354474Cull   /* "STGL" — a Line with no colon in it             */
#define SYS__SETTINGS__FAULT_BYTE   0x53544742ull   /* "STGB" — a Byte that is a space, or not text    */
#define SYS__SETTINGS__FAULT_EMPTY  0x53544745ull   /* "STGE" — an Empty key, value, or interior line  */
#define SYS__SETTINGS__FAULT_TWICE  0x53544744ull   /* "STGD" — a key spelled twice (Duplicate)        */

/* ══ THE UNITS, ONCE, FOR BOTH TIERS ═════════════════════════════════════════════════════════════════
 *
 * ⭐⭐ ONE ROW PER UNIT, AND BOTH READERS EXPAND IT. The scale a key spells is a property of the FORMAT,
 * and this file already carries two implementations of that format by ruling — so a suffix table written
 * out twice would be the one part of the duplication with no reason to exist. A row here is a row for
 * the host and for the card in the same act.
 * ⛳ `_kb` IS KIBIBYTES AND `_mb` MEBIBYTES — 1<<10 and 1<<20, never the decimal ones. Every size in this
 * tree is a power of two (a chunk is 512 nodes of 64 bytes, which is 32 KiB exactly), so a decimal
 * kilobyte would be the one unit that did not divide anything and the arithmetic would stop being exact.
 * ⛳ A FOURTH UNIT IS ONE ROW. ⛔ What it is NOT is a row plus a sweep: the two readers are generated from
 * this, so a unit the card knows and the host does not cannot be written down. ⇒ ★ A LIST CANNOT FORGET;
 * A SWEEP CAN — the same reason `PACKAGE_LIST` and `NN__BUFFER_CLASS_LIST` are lists. */
#define SYS__SETTINGS__UNIT_LIST(X)                                                                     \
    X("_bytes", 0u)                                                                                     \
    X("_kb",   10u)                                                                                     \
    X("_mb",   20u)

#define SYS__SETTINGS__HOST_DONE     0   /* the section is finished, and every line in it was well formed */
#define SYS__SETTINGS__HOST_PAIR     1   /* a key and a value, both spans into the caller's own text      */
#define SYS__SETTINGS__HOST_REFUSED  2   /* the same refusal the card would make, at the same byte        */

/* ⛔⛔ THREE OUTCOMES AND NOT TWO, AND THE THIRD IS THE ONE A TEST HAD TO FIND. This answered a plain
 * bool so "there is no such section" and "that section opens and never closes" were the
 * SAME ANSWER — and the boot, reasonably, read both as "no device_params" and carried on with defaults.
 * ⇒ ★ A TRUNCATED CONFIG FILE BOOTED SILENTLY AT THE WRONG SIZE, which is the exact shape of mistake this
 * format's strictness exists to prevent, arriving through the one function that had no way to say so.
 * ⛳ ABSENT IS LEGITIMATE — a file with no `device_params` is the no-config case and every package uses
 * its default. TRUNCATED IS NOT, and the difference cannot be inferred by a caller. */
#define SYS__SETTINGS__READ_ABSENT     0   /* nothing by that name, which is a fine thing to be      */
#define SYS__SETTINGS__READ_FOUND      1   /* it is there and it reads                               */
#define SYS__SETTINGS__READ_MALFORMED  2   /* the file SAID something and it could not be read       */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_SETTINGS_CUH */
