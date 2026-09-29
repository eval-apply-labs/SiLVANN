#ifndef SILVANN__ENGINE_ABI_BUILDING_CUH
#define SILVANN__ENGINE_ABI_BUILDING_CUH

/* ══ BUILDING A PROGRAM — LISTS, CELLS, PICTURES, BINDINGS, TEXT AND NAMES ════════════════════════════
 * The doors a caller builds a program with before it runs: a list and its cells, a picture of a
 * program and a program back from one, an environment, text in and out, and names in and out. */

/* ⛔ A NAME SPELLED FROM OUT HERE MUST BE SPELLED WITH THE STRING THE SYMBOL TABLE HOLDS. The cell takes no
 * hold, so a string a caller merely made would die under the program and leave its name pointing at
 * whatever the room holds next. These two doors are where a caller's cells enter the machine, so this is
 * where they are asked — one hash and one lookup per spelled name, and nothing for any other cell. */
static __device__ inline bool eng__abi__zzprivate_names_interned(const sys__heap_node* cells, uint64_t count) {
    for (uint64_t i = 0ull; i < count; ++i)
        if (sys__bindings__spelled(cells[i].dtype) && !sys__symbol__interned(cells[i].args[0])) return false;
    return true;
}

/* A string's text — and its sixty-four hash bytes into `hash` when it is not null — copied straight into the
 * caller's buffers, answering the length; all ones when the text does not fit `capacity`, in which case
 * nothing is written. The machine is this process, so there is no room in between to pass through. */
static inline uint64_t eng__abi__zzprivate_copy_string(uint64_t string, unsigned char* text, uint64_t capacity,
                                                       unsigned char* hash) {
    const uint64_t length = sys__string__length(string);
    if (length > capacity || (length != 0ull && text == 0)) return ~0ull;
    const sys__heap_node* at = sys__string__hash_of(string);
    if (hash != 0) memcpy(hash, at, sizeof(sys__heap_node));
    if (length != 0ull) memcpy(text, (const unsigned char*)(at + 1), (size_t)length);
    return length;
}

extern "C" {

/* An empty list, ready to have cells put in it. Zero means it could not be made. */
unsigned long long eng_abi_list_create(void) {
    if (!eng__abi__zzpackage_free_to_launch()) return 0ull;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__list__create());
    return got;
}

/* ⭐ THE WHOLE OF CELL-BUILDING IS THREE NUMBERS, WHICH IS WORTH SAYING BECAUSE IT LOOKS LIKE MORE. A
 * program is written with half a dozen kinds of cell — a verb, a literal, a name being read, a name
 * being quoted, a sublist to evaluate, a list left alone — and every one of them is a node with a KIND,
 * an OPCODE and ONE VALUE, differing only in which fields carry anything. So they collapse into one
 * call and the caller says which by the kind it passes. ⇒ ★ SIX CONSTRUCTORS WERE SIX WAYS OF FILLING
 * THE SAME THREE FIELDS. */
int eng_abi_list_append(unsigned long long list, unsigned int dtype,
                        unsigned long long op_code, unsigned long long value) {
    if (!eng__abi__zzpackage_free_to_launch() || list == 0ull) return 0;
    sys__heap_node cell;
    memset(&cell, 0, sizeof cell);          /* every field, because a node carries more than it is given */
    cell.dtype   = (sys__kind)dtype;
    cell.op_code = op_code;
    cell.args[0] = value;
    bool got = false;
    ENG__ABI__ON_THE_MACHINE(got = eng__abi__zzprivate_names_interned(&cell, 1ull) && sys__sublist__append(list, &cell));
    return got ? 1 : 0;
}

/* ⭐ A CELL, MADE WHERE THE CALLER IS. Every cell a program is written with is a kind, a verb and one
 * value, so making one crosses nothing and costs nothing — and it is what lets a caller write a program
 * without knowing how a node is laid out. */
sys__heap_node eng_abi_cell(unsigned int dtype, unsigned long long op_code, unsigned long long value) {
    sys__heap_node cell;
    memset(&cell, 0, sizeof cell);
    cell.dtype   = (sys__kind)dtype;
    cell.op_code = op_code;
    cell.args[0] = value;
    return cell;
}

/* A whole list in ONE crossing, answered as the cell that names it — so nesting is putting this answer
 * into the cells of the next call. ⚖ ARCHITECT: *"maybe we need to expose a c method called
 * sys__list__create_executable ... which gives us back a sys__heap_node so it can be used in another
 * sublist."*
 * ⭐ WHICH IS WHAT MAKES A PROGRAM CHEAP TO BUILD FROM OUTSIDE: the cost of building one is how many times
 * the door is crossed, and this crosses once per form rather than once per cell.
 * ⛔ IT TAKES NOTHING FROM THE CALLER, on a refusal as well — the package states that contract and this
 * adds nothing to it. A caller that nested one form into another owns BOTH afterwards and gives the inner
 * one back; the outer list holds its own. That is what `append` has always done, and the two now agree. */
int eng_abi_create_executable(const sys__heap_node* cells, unsigned int count, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_free_to_launch() || cells == 0 || count == 0u) return 0;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = eng__abi__zzprivate_names_interned(cells, count)
                                   ? sys__list__create_executable(cells, count) : 0ull);
    if (got == 0ull) return 0;
    answer->dtype   = SYS__KIND__OBJECT_REFERENCE;
    answer->args[0] = got;
    return 1;
}

/* ── A PICTURE OF A PROGRAM, AND A PROGRAM BACK FROM ONE ──────────────────────────────────
 * A picture is a program with nothing writable in it: plain arrays all the way down, so any number of
 * readers hold one at once with nothing coordinating them. Making one costs a walk; reading one costs an
 * addition.
 *
 * ⭐ THEY ARE A PAIR BECAUSE NEITHER END IS USEFUL ALONE FROM OUTSIDE. A caller that can freeze and not
 *   thaw holds something it can never run; one that can thaw and not freeze has no way to come by a
 *   picture. Inside the engine the two ends already have different owners — a procedure is frozen where it
 *   is defined and thawed where it is called — and a caller reaching across this door needs both halves.
 *
 * ⛳ THE ANSWER'S KIND IS THE ONE A PICTURE ALREADY USES FOR AN ARRAY INSIDE IT: counted, and not part of
 *   any list's structure. That is exactly what a picture is to whoever holds one, so the tag is read
 *   rather than invented. */
/* Freeze a program into a picture, answered as the cell that names it — so it goes into another call's
 * cells the way every other answer here does.
 * ⛔ THE PROGRAM IS NOT CONSUMED AND THE PICTURE IS A SECOND THING. A caller holds two afterwards and owes
 * a word about both; a refused freeze hands back nothing at all, so there is no half-owned case.
 * ⛳ A LIST REACHED TWICE, A LIST INSIDE ITSELF, AN EMPTY ONE, OR ONE TOO LONG TO FIT A CHUNK — the package
 * refuses each of them and says which. This answers only that there is no picture; which refusal it was is
 * a question about what was raised, and has a different door. */
int eng_abi_freeze(sys__heap_node program, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_free_to_launch() || program.args[0] == 0ull) return 0;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__sublist__deep_copy_to_node_array(program.args[0]));
    if (got == 0ull) return 0;
    answer->dtype   = SYS__KIND__QUOTED_ARRAY;
    answer->args[0] = got;
    return 1;
}

/* And back: a list of the caller's own, made from a picture the caller is holding.
 * ⛔ THE ANSWER TAKES NO HOLD ON THE PICTURE, which is precisely what lets many callers thaw one picture at
 * the same time — and it is also why the picture is still the caller's to let go of afterwards. Letting go
 * of it while a thaw is running takes the ground out from under that thaw. */
int eng_abi_thaw(sys__heap_node picture, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_free_to_launch() || picture.args[0] == 0ull) return 0;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__list__deep_copy_from_node_array(picture.args[0]));
    if (got == 0ull) return 0;
    answer->dtype   = SYS__KIND__OBJECT_REFERENCE;
    answer->args[0] = got;
    return 1;
}

/* Let go of one hold on something this interface handed out, and answer how many are left. A caller
 * that builds a program per call and never says this keeps every one of them: the allocator has no way
 * to notice that a caller has finished with a list; being told is the only mechanism there is.
 * ⛳ ZERO MEANS IT DIED, which is indistinguishable here from a refusal — both answer nothing is left.
 * A caller that needs to tell them apart is asking whether a fault was raised, which is a different
 * question and has a different door. */
unsigned int eng_abi_release(unsigned long long offset) {
    if (!eng__abi__zzpackage_free_to_launch() || offset == 0ull) return 0u;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = (uint64_t)sys__heap_object__release(offset));
    return (unsigned int)got;
}

/* An environment with room for `symbols` names. `base` is the view a reader shares rather than clones;
 * zero means it has none, which is what an outermost environment has. */
unsigned long long eng_abi_bindings_create(unsigned long long symbols, unsigned long long base) {
    if (!eng__abi__zzpackage_free_to_launch()) return 0ull;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__bindings__create(symbols, base));
    return got;
}

/* How many numbered names an environment holds — the number a composer gives names below and spells names
 * at or above. Zero is a refusal: nothing is an environment of no names. */
unsigned long long eng_abi_bindings_symbols(unsigned long long bindings) {
    if (!eng__abi__zzpackage_free_to_launch() || bindings == 0ull) return 0ull;
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__bindings__symbols(bindings));
    return got;
}

/* The widest an environment can be, which is what a composer with no environment spells names against. */
unsigned long long eng_abi_bindings_width_max(void) {
    return SYS__BINDINGS__WIDTH_MAX;
}

/* ── TEXT, IN AND OUT ────────────────────────────────────────────────────────────────────────────────
 * The bytes are read where the caller holds them, and a string's text goes back out straight into the
 * caller's buffer. A string holds nothing, so reading one is a copy and takes nothing. */
/* A string holding these bytes, answered as the cell that names it. The caller holds it and gives it back
 * like anything else this interface hands out.
 * ⛳ TEXT TOO LONG TO FIT ONE STRING IS STILL HANDED OVER: the package refuses it before it reads a byte, and
 * that refusal is what puts the reason on the fault channel. */
int eng_abi_string_create(const unsigned char* bytes, unsigned long long length, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_free_to_launch() || (bytes == 0 && length != 0ull)) return 0;
    const unsigned char none = 0u;                  /* empty text still hands the package somewhere to look */
    uint64_t got = 0ull;
    ENG__ABI__ON_THE_MACHINE(got = sys__string__create((const uint8_t*)(bytes ? bytes : &none), length));
    if (got == 0ull) return 0;
    answer->dtype   = SYS__KIND__OBJECT_REFERENCE;
    answer->args[0] = got;
    return 1;
}

/* What a string holds: its length, its sixty-four hash bytes and its text. Refused when the reference is
 * not a string, or when `capacity` is smaller than the text — nothing is written past what the caller
 * said it has. `hash` may be null for a caller that does not want it. */
int eng_abi_string_read(unsigned long long string, unsigned char* text, unsigned long long capacity,
                        unsigned char* hash, unsigned long long* length) {
    if (length == 0) return 0;
    *length = 0ull;
    if (!eng__abi__zzpackage_free_to_launch() || string == 0ull) return 0;
    /* ⛳ A REFERENCE THAT IS NOT A STRING ANSWERS ALL ONES, because zero is a length an empty string has. */
    uint64_t got = ~0ull;
    ENG__ABI__ON_THE_MACHINE(if (sys__string__is(string))
                                 got = eng__abi__zzprivate_copy_string(string, text, capacity, hash));
    if (got == ~0ull) return 0;
    *length = got;
    return 1;
}

/* ── NAMES, IN AND OUT ───────────────────────────────────────────────────────────────────────────────
 * A name arrives the way a string's text does and comes back as two words: the number the machine deals
 * it, and the interned string a cell spells it with. A number goes back out as its name's text, straight
 * into the caller's buffer. */
/* The number a name is known by on this machine, dealt the first time it is asked for, and — into `string`
 * when it is not null — the interned string a cell spells the name with. All ones means it was refused,
 * and the fault channel says why; `string` is then zero. */
unsigned long long eng_abi_intern(const unsigned char* bytes, unsigned long long length,
                                  unsigned long long* string) {
    if (string != 0) *string = 0ull;
    if (!eng__abi__zzpackage_free_to_launch() || (bytes == 0 && length != 0ull)) return ~0ull;
    const unsigned char none = 0u;                  /* empty text still hands the package somewhere to look */
    uint64_t got[2] = { ~0ull, 0ull };
    ENG__ABI__ON_THE_MACHINE(got[0] = sys__symbol__intern((const uint8_t*)(bytes ? bytes : &none), length, &got[1]));
    if (string != 0) *string = got[1];
    return got[0];
}

/* A number's name, as text. Refused when nothing was dealt that number, or when `capacity` is too small. */
int eng_abi_symbol_name(unsigned long long symbol, unsigned char* text, unsigned long long capacity,
                        unsigned long long* length) {
    if (length == 0) return 0;
    *length = 0ull;
    if (!eng__abi__zzpackage_free_to_launch()) return 0;
    uint64_t got = ~0ull;
    ENG__ABI__ON_THE_MACHINE(
        const sys__heap_node name = sys__symbol__name(symbol);
        if (name.dtype == SYS__KIND__OBJECT_REFERENCE) {
            got = eng__abi__zzprivate_copy_string(name.args[0], text, capacity, 0);
            (void)sys__heap_object__release(name.args[0]);
        });
    if (got == ~0ull) return 0;
    *length = got;
    return 1;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_BUILDING_CUH */
