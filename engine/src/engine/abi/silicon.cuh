#ifndef SILVANN__ENGINE_ABI_SILICON_CUH
#define SILVANN__ENGINE_ABI_SILICON_CUH

/* ══ THE SILICON FAMILIES ═════════════════════════════════════════════════════════════════════════════
 * Which silicon family the package rooms and the worker are placed on, and how a wheel hands the
 * engine a family it loaded, asks which family a card goes to, and how many devices a family counts. */


extern "C" {

/* ⚖ *"we build for all architectures and depending on which one it is we choose"* — so the family the
 * worker and the package rooms are placed on is chosen here, by its spelling, before a boot; the machine
 * itself is always the host's. With no choice made a boot uses the first family offered, the host's.
 * Answers 1 when a family by that name was offered and 0 when not, and a refused name leaves the earlier
 * choice standing. ⛳ A machine that is already up keeps its worker and rooms where they are: the choice
 * is for the next boot, because moving live rooms to another family is not a thing a name can do. */
int eng_abi_choose_silicon(const char* name) {
    const sys__silicon_family__id family = sys__silicon_family__named(name);
    if (family == SYS__SILICON_FAMILY__NONE) return 0;
    eng__abi__zzpackage_silicon_family_wanted = family;
    return 1;
}

/* ⭐⭐ THE WORKERS THE NEXT BOOT STANDS UP — ⚖ *"go ahead with the multithreading"*: `n` devices, worker `w` the
 * family spelled `names[w]` and its device `devices[w]`, and block `w` of the machine runs as it. Answers 1 when
 * every family was offered and every device is one it counts, 0 otherwise — and a refused list leaves the earlier
 * choice standing. `n` of 0 forgets the list: the next boot has one worker, as `choose_silicon` says. */
int eng_abi_choose_workers(unsigned int n, const char* const* names, const unsigned int* devices) {
    if (n > SYS__SILICON__WORKERS_MAX) return 0;
    sys__silicon_family__id family[SYS__SILICON__WORKERS_MAX];
    for (unsigned int w = 0u; w < n; ++w) {
        family[w] = sys__silicon_family__named(names[w]);
        if (family[w] == SYS__SILICON_FAMILY__NONE || devices[w] >= sys__silicon__device_count(family[w])) return 0;
    }
    for (unsigned int w = 0u; w < n; ++w) {
        eng__abi__zzpackage_wanted_family[w] = family[w];
        eng__abi__zzpackage_wanted_device[w] = devices[w];
    }
    eng__abi__zzpackage_wanted_count = n;
    return 1;
}

/* ⭐ THE CALLER'S THREAD ACTS AS WORKER `w` — its device bound, and every verb it runs through this door's machine
 * reading worker `w`'s context, hatch and settings — until told `-1`, back to worker 0. What a program run from
 * out here uses to load a worker's weights into its own arena and fill its own tables. 1 when it took, 0 for a
 * worker the machine does not have, or while a resident launch is up. */
int eng_abi_act_as(int w) {
    if (!eng__abi__zzpackage_free_to_launch() || w >= (int)eng__abi__zzpackage_worker_count) return 0;
    const unsigned int as = w < 0 ? 0u : (unsigned int)w;
    if (!sys__silicon__bind_device(eng__abi__zzpackage_workers[as].family, eng__abi__zzpackage_workers[as].device)) return 0;
    sys__silicon__host_act_as(w < 0 ? -1 : w);
    return 1;
}

/* How many workers the machine that is up has, and which: worker `w`'s family spelling, and its device. */
unsigned int eng_abi_workers(void) { return eng__abi__zzpackage_up() ? eng__abi__zzpackage_worker_count : 0u; }
const char* eng_abi_worker_silicon(unsigned int w, unsigned int* device) {
    if (!eng__abi__zzpackage_up() || w >= eng__abi__zzpackage_worker_count) return 0;
    if (device) *device = eng__abi__zzpackage_workers[w].device;
    return sys__silicon_family__spelling(eng__abi__zzpackage_workers[w].family);
}

/* ⚖ *"the gpu side becomes a separate binary that is loaded at the start of the cpu session"* — and
 * *"the wheel carries the environment"*: the wheel loads the binary, resolves `silvann_silicon_family_get`,
 * and hands the table it answered here. 1 when its card is now in the table, 0 when it is refused — a
 * version this program does not read, a table too small, or a family that already has a card.
 * ⛳ THE PROGRAM NEVER LOADS A BINARY ITSELF, so it names no loader and no file suffix, and a platform's
 * way of finding a library is the platform's business. */
int eng_abi_offer_silicon_family(const void* family) {
    return sys__silicon_family__offer((const sys__silicon_family*)family) ? 1 : 0;
}

/* What a wheel needs to open a family without naming anything inside the program: the version this
 * program reads, the name of the entry every family exports, and — for a table it was answered — the
 * silicon that table's card is. Null for a table this program does not read. */
unsigned int eng_abi_silicon_family_version(void) { return SYS__SILICON_FAMILY__ABI_VERSION; }
const char* eng_abi_silicon_family_entry(void) { return SYS__SILICON_FAMILY__ENTRY; }
const char* eng_abi_silicon_family_silicon(const void* family) {
    const sys__silicon_family* p = (const sys__silicon_family*)family;
    if (p == 0 || p->abi_version != SYS__SILICON_FAMILY__ABI_VERSION) return 0;
    return p->name;
}

/* ⭐ WHICH SILICON FAMILY A CARD GOES TO, by the card's name — `gfx906`, `sm_86`: offered to every family
 * that includes it, highest number first, and given to the first whose binary opens and sees its cards.
 * `open` is the wheel's — it opens a family's binary and offers its table, answering whether it did — and
 * `devices` counts a family's cards once open. Answers the family's name, or null: a card no family
 * includes is unsupported, and one every family refused says why through the attempts below.
 * ⛳ ONE SESSION PER PROCESS, so a binary is opened once however many cards it takes, and a second family
 * of one vendor is never opened beside the first. ▶ `silicon_families/choose.cuh`. */
static silicon_families__session eng__abi__zzprivate_card_session;
static silicon_families__attempt eng__abi__zzprivate_card_attempts[SYS__SILICON_FAMILY__SLOTS];
static uint32_t                  eng__abi__zzprivate_card_attempt_count;

const char* eng_abi_silicon_family_choose(const char* card, silicon_families__open_fn open,
                                          silicon_families__count_fn devices, void* context) {
    if (card == 0 || open == 0 || devices == 0) return 0;
    const sys__silicon_family__id family = silicon_families__choose(
        silicon_families__rows, (uint32_t)SILICON_FAMILIES__COUNT, &eng__abi__zzprivate_card_session, card,
        open, devices, context, eng__abi__zzprivate_card_attempts, SYS__SILICON_FAMILY__SLOTS,
        &eng__abi__zzprivate_card_attempt_count);
    return silicon_families__name(family);
}

/* How many families the last card was offered to, and — for each, in order — its name and what came of
 * it, as a sentence. Null past the last. */
unsigned int eng_abi_silicon_family_attempts(void) { return eng__abi__zzprivate_card_attempt_count; }
const char* eng_abi_silicon_family_attempt(unsigned int i, const char** outcome) {
    if (i >= eng__abi__zzprivate_card_attempt_count) return 0;
    const silicon_families__attempt* a = &eng__abi__zzprivate_card_attempts[i];
    if (outcome != 0)
        *outcome = (a->outcome == SILICON_FAMILIES__TAKEN)          ? "taken"
                 : (a->outcome == SILICON_FAMILIES__WONT_OPEN)      ? "its binary would not open here"
                 : (a->outcome == SILICON_FAMILIES__SEES_NONE)      ? "it opened and sees no card of its own"
                 : "another family of its vendor is open in this process — this card needs a process of its own";
    return silicon_families__name(a->family);
}

/* How many devices a silicon has, counted by its family; 0 for one with no family loaded. */
unsigned int eng_abi_devices(const char* name) {
    const sys__silicon_family__id family = sys__silicon_family__named(name);
    return (family == SYS__SILICON_FAMILY__NONE) ? 0u : sys__silicon__device_count(family);
}

/* The spelling of the family the worker and the package rooms are on — or, with nothing up, the one a
 * boot would place them on. Null when no family was offered at all. */
const char* eng_abi_silicon(void) {
    const sys__silicon_family__id family = eng__abi__zzpackage_up() ? eng__abi__zzpackage_worker_silicon_family
        : (eng__abi__zzpackage_silicon_family_wanted != SYS__SILICON_FAMILY__NONE) ? eng__abi__zzpackage_silicon_family_wanted
        : sys__silicon_family__first();
    return sys__silicon_family__of(family) ? sys__silicon_family__spelling(family) : 0;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_SILICON_CUH */
