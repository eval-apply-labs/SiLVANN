#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PACKAGE_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PACKAGE_CUH

/* This file needs nothing: every name in it is its own. */

/* ── HOW MUCH ROOM A PACKAGE OWNS, STATED BY THE PACKAGE ─────────────────────────────────────────────
 *
 * Some packages need memory that is not the heap's: an arena, a table, a cache — anything a node in the
 * tree can NAME while its bytes live somewhere else entirely. Only the package knows what that is, so
 * only the package states the figure, in its own file, behind a guard so a build can override it.
 *
 * ⭐⭐ THE PACKAGE TAKES IT AND THE PACKAGE OWNS IT. ⚖ *"one allocation per package"*: a figure that comes
 * out of a config file is known only to the package that reads it, so no engine-wide sum of them can
 * exist. So each package's HOST INIT takes its own, and the engine keeps a record it never reads.
 * ⛳ THE ENGINE NEVER LEARNS WHOSE BYTES ARE WHOSE. Ownership is about who decides what goes in it, not
 * about who called the allocator — and the same package does both.
 * ⛳ AND THE SLICE ARRIVES THROUGH THE EXTENSION HATCH, which is what that register row was designed for:
 * ⚖ *"the package can find its index and have an entry point without polluting the register."* One entry
 * per package, indexed by the package id, which a package always knows because it is its own.
 *
 * ⛳ EVERY PACKAGE DEFINES ONE EVEN AT ZERO, for the reason every uniform list has: the name is PASTED
 * from the package's, so the gather names it whether or not the package wants any room. Zero is an
 * answer — it means the hatch entry stays empty and nothing is taken on that package's behalf.
 * ⛔ `sys` IS ZERO AND WILL STAY THERE. Its memory IS the heap, and the boot takes that already; a second
 * span for the language would be a second allocator nobody asked for. */
#ifndef sys__PACKAGE__BYTES
#define sys__PACKAGE__BYTES 0ull
#endif

/* A package's host init answers whether it stood up. ⛔ A REFUSAL MUST LEAVE NOTHING TAKEN — it is the
 * one path where the unwind cannot help, because the loop only tears down the inits that SUCCEEDED. */
#define SYS__PACKAGE__ROOM_NONE  { 0, 0ull }

/* ══ THE HATCH IS TWO LEVELS — `hatch[package_id][row]` ══════════════════════════════════════════════
 *
 * ⚖⚖ RULED: *"one system register field that is called package_internal and references an
 * array, where each package has its own sub array. this would make finding values a
 * system_register[PACKAGE_ID][SETTING_ENUM_ID]"* — and, on a package that wants none: *"a package with no
 * system registers will have a null as system_register[package_id] so we can still find thing by index
 * with the package enum. the hole buys us direct referenceability."*
 *
 * ⭐ THE TOP LEVEL ALREADY EXISTED AND THE HOLE IS WHAT MAKES IT WORK. The array is `package_count` long
 * and a package's id IS its index, so a package with nothing reserved is a NULL ENTRY rather than an
 * absence — the array is never compacted and no lookup ever has to search. ⇒ ★ A HOLE IS CHEAPER THAN A
 * SEARCH, and it is what keeps `own_row` two loads rather than a walk.
 *
 * ⛔⛔ THE ALTERNATIVE — ONE FLAT LIST WITH EVERY PACKAGE'S ROWS IN IT — IS RULED OUT, AND IT WAS RULED
 * OUT BEFORE THIS FILE ASKED. ▶ `system_register__header.cuh`: *"a list that merged every package's rows
 * would deal them in roster order, so one package adding a row moves every seat belonging to every
 * package after it — and a program written against the old numbering then reads the wrong row with
 * nothing failing to build."* A row's number is something a composed program can hold. Two levels is what
 * lets `sys` move only `sys`'s seats.
 *
 * ── WHO OWNS WHICH ROW ──────────────────────────────────────────────────────────────────────────────
 * ROW 0 IS THE ROOM AND IT IS `sys`'s, because the engine is what writes it: a package's host init takes
 * its span, and the stand-up's runner zero writes the address into the hatch, so that crossing belongs
 * to the boot and not to the package. A PACKAGE'S OWN ROWS BEGIN AT 1 and are numbered by its own enum,
 * from its own list, so a row `sys` adds can never move one of nn's.
 * ⇒ ⭐ WHICH IS WHY THE SPAN IS NOT A SPECIAL CASE: everything device-wide a package owns is
 * `hatch[pkg][row]`, and the room is simply the first row.
 *
 * ── ⛳ A HATCH ROW IS PERENNIAL, AND THAT IS WHY READING ONE COSTS NO ATOMIC ─────────────────────────
 * Both hops BORROW rather than `get`. `get` puts an atomic on an element's count for every look, and a
 * release path that paid one per buffer freed would be an atomic in the hottest loop this package will
 * have. Borrowing is safe exactly where `get`'s hold has no work to do — *"an array nothing writes, held
 * by whoever reads it, cannot lose one"* — so the premise is stated rather than assumed:
 *     `REASONED`: a hatch row is written at STAND-UP, before the engine runs, and not again. The top
 *     array is written only by `zzengine_open_entry`, the room only by `zzengine_hand_room`, and both run
 *     on the stand-up's runner zero, single threaded, before any init is evaluated. A package's own row
 *     is written
 *     by its OPCODE init, which is also before any program runs.
 * ⛔ WHAT WOULD FALSIFY IT: anything that REWRITES a row while the machine is up. Such a row must be read
 * through `get` and its hold given back, and it would be the first of its kind here. The register's own
 * `SETTINGS` row is already documented as writable *"if someone wants to load a new config"*, so this is
 * a property of how the hatch is used and not a guarantee the storage makes.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

#define SYS__PACKAGE__HATCH_AT     0u   /* args[0] of the ROOM row: where the room starts */
#define SYS__PACKAGE__HATCH_BYTES  1u   /* args[1]: how much of it there is               */
#define SYS__PACKAGE__HATCH_RAM_AT      2u   /* args[2]: the RAM span, as the program reads it   */
#define SYS__PACKAGE__HATCH_RAM_CARD_AT 3u   /* args[3]: ...as a kernel writes it                */
#define SYS__PACKAGE__HATCH_RAM_BYTES   4u   /* args[4]: ...and how much, 0 for none             */

#define SYS__PACKAGE__HATCH_ROOM      0u   /* row 0 of every sub-array — `sys`'s, written by the boot */
#define SYS__PACKAGE__HATCH_OWN_FIRST 1u   /* where a package's own enum starts                       */
#define SYS__PACKAGE__DECLARE_HATCH_ROWS(PKG)                                                           \
    enum {                                                                                              \
        PKG##__HATCH__ZZBASE = (int)SYS__PACKAGE__HATCH_OWN_FIRST - 1,                                  \
        PKG##__HATCH_ROW_LIST(SYS__PACKAGE__HATCH_ROW)                                                  \
        PKG##__HATCH__END                                                                               \
    };                                                                                                  \
    enum { PKG##__HATCH__ROWS = (int)PKG##__HATCH__END - (int)SYS__PACKAGE__HATCH_OWN_FIRST };

/* ⛔ `sys` DECLARES NONE, AND IT IS THE SAME ARGUMENT AS ITS ZERO SPAN: its device-wide state is the
 * register's own rows, which it owns outright and reaches without a hatch. A row here would be `sys`
 * going through a door built so that it would not have to open one for everybody else.
 * ⛳ SO `sys` IS THE HOLE — entry 0 of the hatch is null in every booted machine, which makes the case the
 * ruling asked for the one the suite exercises by default rather than a path nothing takes. */
#define sys__HATCH_ROW_LIST(X)

/* How that array is laid out: five words per package — `at`, `bytes`, then the RAM span's `ram_at`,
 * `ram_card_at`, `ram_bytes` — in roster order. ⛔⛔ IT IS
 * DECLARED ONCE, HERE, BECAUSE THE WRITER AND THE READER ARE IN DIFFERENT TIERS. The engine fills the
 * array from what the host inits took, and this package reads it in `zzengine_fill_hatch`; a stride
 * written down twice is two numbers
 * obliged to agree with nothing checking them, and the symptom of their disagreeing is every package
 * after the first one being handed somebody else's address. */
#define SYS__PACKAGE__ROOM_SLICE 5ull

/* Why an init refused. Separate words because each has a different repair, and a caller that only learns
 * "no" has to guess which. */
#define SYS__PACKAGE__FAULT_ARITY  0x504B4741ull   /* "PKGA" — an init called with something other than the list */
#define SYS__PACKAGE__FAULT_LIST   0x504B474Cull   /* "PKGL" — the argument is not a list to walk           */
#define SYS__PACKAGE__FAULT_ROOM   0x504B4752ull   /* "PKGR" — a deferral could not be appended            */
#define SYS__PACKAGE__FAULT_HATCH  0x504B4748ull   /* "PKGH" — the hatch would not stand up or take a slice */

/* ⭐ A PACKAGE DECLARES ITS ROWS THE WAY IT DECLARES EVERYTHING ELSE — a list, gathered, with no number
 * typed anywhere. The enum starts one below `HATCH_OWN_FIRST` so the first row lands on it, and the count
 * falls out of where the list ended. An EMPTY list gives a count of zero, which is a package that wants
 * no state of its own and is the thing the hole is for. */
#define SYS__PACKAGE__HATCH_ROW(NAME)  NAME,
SYS__PACKAGE__DECLARE_HATCH_ROWS(sys)

#include <stdint.h>
#include "../abi/silicon_family.cuh"   /* what a family is, which a room is taken on */

/* ── THE HOST HALF OF STANDING A PACKAGE UP ──────────────────────────────────────────────────────────
 *
 * ⚖ ARCHITECT: *"make the allocator method an abi one, by looping through the packages and
 * call their init method… each init method has a reference to the file and can set up their own area,
 * and then order dependent setup can be done in the opcode by reading the dictionary to get the values."*
 *
 * ⭐⭐ SO A PACKAGE HAS TWO INITS, AND THE RULE THAT DIVIDES THEM IS THE ARCHITECT'S:
 * ⚖ *"allocations need to not be order dependent and order dependent code is in the opcode."*
 * ```
 *   THE HOST INIT (here)     reads the config, takes ITS OWN allocation, sets up ITS OWN area.
 *                            Runs in a plain loop over the roster, BEFORE the machine is up.
 *   THE OPCODE INIT          reads the settings dictionary and does what needs a standing engine.
 *                            Runs off the init list, which sorts itself by DEFERRAL.
 * ```
 * ⛔⛔ AND THE LOOP HAS AN ORDER — THE ROSTER'S — WHICH IS WHY THE RULE IS A RULE AND NOT A PROPERTY. A
 * host init that read what another package's host init had set up would NOT fail: it would work, for
 * exactly as long as the roster stayed in the lucky order, and break the day somebody adds a package.
 * ⚖ S104 made package init a program precisely to end that (*"the list is the readiness state, so a
 * roster row's position now decides NOTHING"*), and a plain loop puts position back in charge one launch
 * earlier. ⇒ ★ A MECHANISM THAT REFUSES IS SAFE; ONE THAT SUCCEEDS BY LUCK IS NOT — so:
 *     A HOST INIT TOUCHES ONLY ITS OWN. Its config, its allocation, its area. It does not read another
 *     package's area, its hatch entry, or anything that package's host init wrote. Anything that needs
 *     to happen after somebody else is the OPCODE init's, which is the side with a readiness state.
 *
 * ── WHY A TEARDOWN SITS BESIDE IT RATHER THAN BEING ADDED LATER ─────────────────────────────────────
 * A package that takes its own room gives its own back, so the ABI never holds an array of other
 * people's pointers. ⛔ AND THE REAL REASON IS THE FAILURE PATH: a LOOP TURNS ONE FAILURE POINT INTO N.
 * If package k's host init refuses, packages 0..k-1 have already taken memory, and the boot has to
 * unwind in reverse over exactly the ones that succeeded.
 * ⛳ `MEASURED`, because it decides how defensive this has to be: a dead process' VRAM comes
 * back — 8 GiB taken, `SIGKILL`, and free VRAM returns to the byte. So the crash path is safe by
 * construction and nothing here has to survive one. What is NOT safe is the two paths inside a LIVING
 * process: `shutdown`, and the failed boot above. ⇒ the reclaim NARROWS this contract rather than
 * retiring it.
 * ⛔ AND THE FIGURE THAT SIZES IT IS SMALLER THAN IT FIRST LOOKED, corrected the same day: a boot costs
 * 4 MiB at the default geometry, not the 172 MiB the FIRST boot appears to cost — 168 of those are the
 * HIP context, taken once per process and returnable only by exiting. The give-back is still what keeps
 * eleven cycles flat; it is just keeping 4 MiB flat and not 172. A config asking for six gigabytes is
 * what makes the number matter.
 * ▶ `measurements/2026-09-18_device_memory_on_process_death.md`.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What a host init took, and what its teardown gives back. The ABI carries one per package, writes the
 * family before the init runs and reads neither of the others — they are the package's own record, handed
 * back to the package. */
typedef struct {
    sys__silicon_family__id family;  /* the silicon to take it on: the machine's, named by the engine   */
    void*    at;      /* what was taken from the host, or null when the package asked for none */
    uint64_t bytes;   /* how much of it there is, which is what the hatch entry will say       */
    /* A second span, in pinned RAM the card can write (`pin_mapped`): where the program reads it, where a
     * kernel writes it, and how much. All zero when the package asked for none. */
    void*    ram_at;
    void*    ram_card_at;
    uint64_t ram_bytes;
} sys__package_room;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_PACKAGE_CUH */
