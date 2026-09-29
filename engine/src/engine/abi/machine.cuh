#ifndef SILVANN__ENGINE_ABI_MACHINE_CUH
#define SILVANN__ENGINE_ABI_MACHINE_CUH

/* ══ THE MACHINE — STANDING IT UP AND TAKING IT DOWN ══════════════════════════════════════════════════
 * The one machine and the scratch its answers come back through, the resident flag, the package
 * rooms and the program that stands the packages up; and the doors that boot, shut down, report a
 * boot's status and its geometry. Everything else in `abi/` is written against what this file defines. */

/* ── THE ONE MACHINE, AND THE SCRATCH EVERY ANSWER COMES BACK THROUGH ────────────────────────────────
 * A launch of several runners answers from a thread that is not the caller's, so its answer is written
 * to a node the machine owns and read back once the runners are joined. One node serves every such door:
 * each joins its runners before it reads, and none is called while another is running. A node is the
 * widest thing ever written there, so a node is what it is. A door on one runner answers into its own
 * locals and needs none of this. */
/* ⛳ ZERO-INITIALISED BY EMPTY BRACES RATHER THAN BY POSITION, because a positional list is a second
 * statement of how many fields the handle has — and when one left, this was the only place that still
 * believed the old count. The compiler caught it, which is the good case; a shorter list would have
 * been accepted in silence. Empty braces name no field and state no count, so there is nothing here
 * that can fall behind the handle. */
static EngineBoot   eng__abi__zzpackage_machine = {};
/* The family the next boot places the worker and the package rooms on, by choice; `SYS__SILICON_FAMILY__NONE`
 * leaves it to the first family offered, which is the host's — it is compiled in and offers itself before
 * a wheel can offer one it loaded. ▶ `eng_abi_choose_silicon`. */
static sys__silicon_family__id eng__abi__zzpackage_silicon_family_wanted = SYS__SILICON_FAMILY__NONE;
/* ⭐ THE WORKER'S SILICON, WHICH IS NOT THE MACHINE'S. The machine — heap,
 * counters, fault channel, every buffer the evaluator reads — is the HOST's, because the evaluator runs
 * there and dereferences them; the packages' rooms and the verbs' doors are on the silicon chosen here,
 * device 0 of it. The fault word the verbs launch with is
 * on the worker's device, taken at the boot and given back at shutdown. */
static sys__silicon_family__id eng__abi__zzpackage_worker_silicon_family = SYS__SILICON_FAMILY__NONE;   /* worker 0's */

/* ⭐⭐ AND THE WORKERS — ⚖ *"go ahead with the multithreading"* · *"a disjointed setup with one card holding the kv
 * cache, another one doing the fixed compute and the cpu doing the sparse experts"*. A worker is one device of
 * one family — a card, or the CPU — and block `w` runs as worker `w` (`sys__silicon__worker`). Each has its own
 * fault word, its own heap view, and its own package rooms, taken by each package's host init from that
 * worker's own settings. With none chosen a boot has one: the chosen family's device 0, as before. */
typedef struct {
    sys__silicon_family__id family;
    uint32_t                device;
    uint32_t*               fault_word;   /* what the worker's doors flag an overflow in */
    void*                   heap_card;    /* the heap in the worker's view, or 0 when it could not register it */
} eng__abi__worker;
static eng__abi__worker        eng__abi__zzpackage_workers[SYS__SILICON__WORKERS_MAX];
static unsigned int            eng__abi__zzpackage_worker_count = 0u;              /* the booted machine's */
static sys__silicon_family__id eng__abi__zzpackage_wanted_family[SYS__SILICON__WORKERS_MAX];
static uint32_t                eng__abi__zzpackage_wanted_device[SYS__SILICON__WORKERS_MAX];
static unsigned int            eng__abi__zzpackage_wanted_count = 0u;              /* 0: one worker, chosen as before */
static sys__heap_node*  eng__abi__zzpackage_answer  = 0;

/* ── THE WAY IN WHILE SOMETHING IS ALREADY RUNNING ───────────────────────────────────────────────────
 * While runners are working, a door that would run work of its own is refused (below), so talking to
 * them needs a path that runs nothing: the side channel, whose writes are fenced release and whose reads
 * are fenced acquire, so a runner that sees a status sees the words sent before it. ▶
 * `silicon_families/host/sys.cuh`. The landing room is the node those words are staged in on the way out.
 * On the host `sys__gpu__compute_completed()` answers true having done nothing, because everything asked
 * for has finished by the time the asking returns.
 * ⛳ THEY ARE HELD WITH THE MACHINE AND NOT TAKEN PER CALL, unlike the two words a dispatch borrows. A
 * channel is a property of being up rather than of one question, and a door used while runners are
 * working then allocates nothing. */
static void*           eng__abi__zzpackage_channel = 0;
static sys__heap_node* eng__abi__zzpackage_landing = 0;

/* ── AND WHETHER THERE IS A LAUNCH OUT THERE THAT IS NOT GOING TO END ────────────────────────────────
 * ⭐⭐ EVERY OTHER DOOR IN `abi/` RUNS WORK ON THE MACHINE, AND WHILE A RESIDENT LAUNCH IS UP EVERY BLOCK
 *   ALREADY HAS A RUNNER. A door that ran work then would be a second runner zero, or a second pool over
 *   blocks already taken — carving from arenas whose own runners are live and resetting the runner count
 *   they read. `REASONED` from `engine/launch.cuh`: nothing detects that race, so the flag exists to turn
 *   it into a refusal, and that is its whole job.
 * ⛳ IT GATES READING AS WELL AS RUNNING. Reading a string's text or a name runs `sys` on a runner the
 *   same way, and is refused the same way. What stays open is what runs nothing: the side channel — the
 *   three scheduling doors — and `sys`'s own looking-inside doors such as `sys_abi_peek`, i.e. the entire
 *   vocabulary a caller needs while the engine is working.
 * ⛔ A FAULT IS STILL VISIBLE, WHICH IS WHY REFUSING TO READ THE CHANNEL IS AFFORDABLE: a computation
 *   that failed leaves its base at ERROR with the word in the result row, and `poll` answers both. */
static int             eng__abi__zzpackage_resident = 0;
static unsigned int*   eng__abi__zzpackage_stop     = 0;

static inline bool eng__abi__zzpackage_up(void) {
    return eng__abi__zzpackage_machine.heap != 0 && eng__abi__zzpackage_answer != 0
        && eng__abi__zzpackage_channel != 0     && eng__abi__zzpackage_landing != 0;
}

static inline bool eng__abi__zzpackage_free_to_launch(void) {
    return eng__abi__zzpackage_up() && eng__abi__zzpackage_resident == 0;
}

/* Tell a resident launch to end, and wait for it. ⛔ IT IS A HELPER AND NOT ONE OF THE DOORS BECAUSE
 * SHUTTING THE ENGINE DOWN OWES THE SAME ACT: giving the pool back while its runners are still reading
 * out of it is not a leak, it is a runner walking freed memory. ⛳ AND IT REPORTS FAILURE BY CHANGING
 * NOTHING — if the word did not go, the launch is still out there and saying otherwise would leave the
 * caller believing it may take the memory back. */
static inline bool eng__abi__zzpackage_stand_down(void) {
    if (eng__abi__zzpackage_resident == 0) return true;
    /* The stop word goes out from the landing room, like every word the side channel sends; what it
     * holds between calls is nobody's. */
    *(unsigned int*)&eng__abi__zzpackage_landing->args[0] = 1u;
    if (!sys__gpu__side_memory_write(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop,
                                        &eng__abi__zzpackage_landing->args[0],
                                        sizeof(unsigned int), eng__abi__zzpackage_channel))
        return false;
    if (!sys__gpu__compute_completed(eng__abi__zzpackage_machine.family)) return false;
    /* ⛳ THE WAIT IS THE JOIN. The stop word is written, so every runner leaves its park once it has
     * finished whatever it is running, and this returns when the last one has — as long as the longest
     * computation still in flight and not one instant longer. The stop word is given back only after,
     * because a runner still reading it would read freed memory. */
    if (!eng__launch__join_resident()) return false;
    sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop);
    eng__abi__zzpackage_stop     = 0;
    eng__abi__zzpackage_resident = 0;
    return true;
}

/* ── THE WORK, ONE KERNEL PER VERB ───────────────────────────────────────────────────────────────────
 * ⛳ ONE KERNEL EACH RATHER THAN ONE KERNEL WITH A SELECTOR. A selector would be an interpreter living
 * inside the interface to the interpreter, and it would put a branch nobody asked for in front of every
 * call. A small entry point costs one name and reads as what it does, however many of them there are.
 *
 * ⛳ AND THE BLOCK'S COMPUTING BASE IS FOUND THE WAY THE ALLOCATOR FINDS IT — by asking the register
 * where the bases are and adding the block's own number. The engine holds no table of its own and never
 * the meaning of an offset; a copy of either here would be free to disagree with the one place it is
 * defined, and the register is now that place. */
/* ⛳ IT TAKES WHOSE RATHER THAN READING IT, and the reason is that asking for somebody else's is not an
 * exotic case: scheduling work onto a block is done BY filling that block's base, so a scheduler names a
 * base that is not its own every time it dispatches. A lookup that read the block id for itself could
 * only ever answer one of them. */
static __device__ inline sys__heap_node* eng__abi__zzpackage_base(uint32_t whose) {
    const sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
    if (where.dtype != SYS__KIND__VALUE_INT || where.args[0] == 0ull) return (sys__heap_node*)0;
    const uint64_t at = sys__heap__object_full_address(where.args[0] + (uint64_t)whose)->args[0];
    return (at == 0ull) ? (sys__heap_node*)0 : sys__heap__object_full_address(at);
}

/* ── STANDING THE PACKAGES UP, WHICH IS A PROGRAM AND NOT A CALL SEQUENCE ────────────────────────────
 *
 * ⚖⚖ ARCHITECT: *"init therefore takes the init of all the packages and builds it as a lisp procedure
 * and executes it."* The boot stands the LANGUAGE up in C, because nothing can run before it; the
 * packages then stand themselves up in a machine that is already whole, which is what makes the ordering
 * theirs to decide rather than the roster's.
 *
 * ⭐⭐ AND THE LIST IS THE READINESS STATE. A call still in it has not run; one that has is gone. The
 * whole of the dependency question is answered by that, with no flag to publish and no flag to read —
 * the reasoning is in `packages/sys/cpu/package__header.cuh`, which is also where the three operations an
 * init is written out of live.
 *
 * ⛳ IT IS HERE RATHER THAN IN `boot.cuh` FOR ONE CONCRETE REASON: evaluating needs a computing base, and
 * the accessor for one is in this file. `boot.cuh` is read before it and the package's own accessor is
 * package-private, so the driver would have to duplicate the lookup to sit there — a second copy of an
 * address calculation, which is the kind of thing that is right until one of them moves.
 *
 * ⛳ THE FORM IS BUILT PER TURN. `(init <list>)` is made, evaluated and let go for each call, so the cell
 * naming the list is never stored INSIDE the list — a list holding a reference to itself is a count that
 * can never reach zero, and nothing here collects one.
 *
 * ⚠ THE BOUND IS NOT THE PROGRESS DETECTOR AND DOES NOT PRETEND TO BE. ⚖ The detector — a full pass that
 * completes no init — is DEFERRED by ruling, to be built when there are enough packages for a cycle to
 * exist. What is here is a plain ceiling on turns, so that a mistake in a deferral is a refusal naming
 * what was still pending rather than a boot that spins forever. Against `package_count` packages a
 * correct list finishes in exactly that many turns; the ceiling is far above any legitimate deferral
 * pattern and exists only to make the failure sayable. */
#define ENG__ABI__PACKAGE_INIT_TURNS  (64u * (uint32_t)package_count)

/* ── THE ROOM THE PACKAGES ASKED FOR ─────────────────────────────────────────────────────────────────
 * ⚖ RULED: *"make the allocator method an abi one, by looping through the packages and call their init
 * method… each init method has a reference to the file and can set up their own area."* So it is ONE
 * ALLOCATION PER PACKAGE, each taken by the package that wanted it in its host init, sized from the
 * config it was handed — a figure only knowable after reading the file — and the engine adds nothing up.
 * ⛳ WHAT THE ENGINE KEEPS IS A RECORD IT DOES NOT READ: one `sys__package_room` per package, written by
 * that package's host init and handed back to that package's teardown. The ABI never learns whose bytes
 * are whose. */
static sys__package_room eng__abi__zzprivate_rooms[SYS__SILICON__WORKERS_MAX][(size_t)package_count];
static unsigned int   eng__abi__zzprivate_rooms_taken[SYS__SILICON__WORKERS_MAX];   /* how many host inits SUCCEEDED, a worker */

/* Each package's host init for worker `w`, in roster order, over that worker's own config text. `taken` is
 * what the unwind walks back over, so it counts only the ones that answered true — a package that refused
 * took nothing and must not be torn down. */
#define ENG__ABI__HOST_INIT_ROW(ID, NAME, ...)                                                          \
    do { if (ok) {                                                                                      \
        ok = NAME##__package__host_init(wconfig, wconfig_length, &eng__abi__zzprivate_rooms[w][(ID)]);   \
        if (ok) eng__abi__zzprivate_rooms_taken[w] = (unsigned int)(ID) + 1u; } } while (0);

/* And the teardown, reached BY ID rather than by a reversed list — a macro cannot be expanded backwards,
 * and the unwind has to run in reverse over a count only known at runtime. ⛳ So the loop is ordinary C
 * over the ids and this switch is what turns one into the right package's verb. */
#define ENG__ABI__HOST_TEARDOWN_CASE(ID, NAME, ...)                                                     \
    case (ID): NAME##__package__host_teardown(&eng__abi__zzprivate_rooms[w][(ID)]); break;
static void eng__abi__zzprivate_give_rooms_back(void) {
    /* ⛳ THE LAST WORKER FIRST, AND EACH ON ITS OWN DEVICE: a room is given back where it was taken, so the
     *   device is bound before its packages tear down. Worker 0's is bound again last, the caller's. */
    for (int w = (int)SYS__SILICON__WORKERS_MAX - 1; w >= 0; --w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        if (eng__abi__zzprivate_rooms_taken[w] == 0u && me->fault_word == 0) continue;
        if (me->family != SYS__SILICON_FAMILY__NONE) (void)sys__silicon__bind_device(me->family, me->device);
        for (int id = (int)eng__abi__zzprivate_rooms_taken[w] - 1; id >= 0; --id) {
            switch (id) { PACKAGE_LIST(ENG__ABI__HOST_TEARDOWN_CASE) default: break; }
        }
        eng__abi__zzprivate_rooms_taken[w] = 0u;
        /* The worker's fault word is on the same device as its rooms and goes back with them, on every road
         * out — a refused boot as much as a shutdown — and its context stops naming a device it gave up. */
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED   /* a harness that brings its own context binds it itself */
        sys__engine__ctx_bind((unsigned int)w, SYS__SILICON_FAMILY__NONE, 0u, 0);
#endif
        if (me->fault_word != 0) sys__gpu__memory_free(me->family, me->fault_word);
        me->fault_word = 0;
    }
    if (eng__abi__zzpackage_workers[0].family != SYS__SILICON_FAMILY__NONE)
        (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[0].family, eng__abi__zzpackage_workers[0].device);
}
#undef ENG__ABI__HOST_TEARDOWN_CASE

/* What the stand-up is handed so it can fill the hatch: one slice per package, in roster order. ⛳ THE
 * HOST INITS HAVE THE ADDRESSES, AND THE HATCH IS FILLED BY RUNNER ZERO INSIDE THE STAND-UP — after the
 * register exists and before the first init reads its room — so they cross as an argument of that launch
 * rather than being written from here, which is the same reason the settings text does. */
/* ⛳ THE LAYOUT IS `sys__package__zzengine_fill_hatch`'s TO STATE, because it is the reader — ▶
 * `SYS__PACKAGE__ROOM_SLICE`. Naming it again here would be a second writable copy of one fact. */


/* ⛔ THE BASE COLUMN IS NOT NAMED HERE, AND THAT IS A RULE RATHER THAN TIDINESS. A row carries a base
 * and the package is applied to it once, at the gather; an arm that reads the raw column is one edit away
 * from using an untagged id, which is how thirty-three opcodes once escaped their package. What this arm
 * wants is the TAGGED constant, which the row already carries as its own name. */
#define ENG__ABI__INIT_CALL(NAME)                                                                       \
    do { if (ok) { sys__heap_node call;                                                                 \
                   call.dtype = SYS__KIND__STANDARD; call.num_args = 0u;                                \
                   call.op_code = NAME; call.args[0] = 0ull;                                            \
                   ok = sys__sublist__append(list, &call); } } while (0);
#define ENG__ABI__INIT_GATHER(ID, NAME, ...)  NAME##__LANGUAGE_CONTRACT__VERBS(ENG__ABI__INIT_ROW, NAME)

/* Only the init verbs go in the list, and they are told apart by their spelling — the one column a row
 * carries that says what a word IS rather than what it does. A package publishing no init contributes
 * nothing here and is not in the list at all, which is the same thing as having already run. */
#define ENG__ABI__INIT_ROW(PKG, ZZBASE, SPELLING, APPLY, NAME, ...)                                     \
    if (eng__abi__zzprivate_is_init(SPELLING)) ENG__ABI__INIT_CALL(NAME)

static __device__ inline bool eng__abi__zzprivate_is_init(const char* spelling) {
    /* The suffix is `__package__init`, matched from the end so a package's own name never has to be
     * known here. Written out, because the C library has no one call for a suffix. */
    static const char suffix[] = "__package__init";
    uint32_t n = 0u, m = 0u;
    while (spelling[n] != '\0') ++n;
    while (suffix[m]   != '\0') ++m;
    if (n < m) return false;
    for (uint32_t i = 0u; i < m; ++i) if (spelling[n - m + i] != suffix[i]) return false;
    return true;
}

static __global__ ENG_BOOT_BLOCK void eng__abi__k_stand_up_packages(const uint64_t* slices,
                                                                    const unsigned char* settings_text,
                                                                    const uint64_t* settings_lengths,
                                                                    int* status) {
    if (sys__silicon__block_id() != 0u) return;
    sys__heap__forget_base();

    /* ⛳ THE HATCH FIRST, AND BEFORE THE LIST IS EVEN BUILT. An init reads its own room, so the table has
     * to be standing and filled before the first one can run — and a package that finds nothing there
     * cannot tell "not handed out yet" from "not asked for", which is the one ambiguity this ordering
     * removes rather than documents.
     * ⛔ THE SLICES ARE READ, NOT COMPUTED. Each package took its OWN allocation in its host init, so there
     * is no common base to add offsets to — this copies what the host recorded into the entries. */
    sys__heap_node* here = eng__abi__zzpackage_base(0u);
    bool ok = (here != 0) && sys__package__zzengine_publish_hatch(package_count);
    /* ⭐ WHO GETS AN ENTRY AND HOW LONG IT IS ARE THE HATCH'S RULES, so the loop that applies them lives
     * with the hatch. What the engine contributes is the addresses and a runner to write them with — the
     * two things a package cannot do for itself. ⛳ AND THE RULE IS THEREFORE TESTABLE WITHOUT A BOOT. */

    /* ⭐⭐ AND EVERY WORKER'S, ONE AFTER ANOTHER, FROM THIS RUNNER ACTING AS EACH IN TURN — its device bound,
     *   its hatch filled from its own slices, its settings parsed from its own section, and every package's
     *   init run for it. The settings row is an array of them, standing before the first init reads one. */
    const uint64_t workers = (uint64_t)sys__silicon__worker_count();
    const uint64_t all_settings = ok ? sys__node_array__create(workers) : 0ull;
    ok = ok && all_settings != 0ull
            && sys__system_register__set(SYS__SYSTEM_REGISTER__SETTINGS, SYS__KIND__OBJECT_REFERENCE, all_settings);
    uint64_t text_at = 0ull;
    for (uint64_t w = 0ull; ok && w < workers; ++w) {
        sys__silicon__host_act_as((int)w);
        ok = sys__silicon__bind_device(eng__abi__zzpackage_workers[w].family, eng__abi__zzpackage_workers[w].device);
        if (ok) ok = sys__package__zzengine_fill_hatch(slices + w * (uint64_t)package_count * SYS__PACKAGE__ROOM_SLICE);
        const unsigned char* settings_here = settings_text + text_at;
        const uint64_t settings_length = settings_lengths[w];
        text_at += settings_length;

        /* ⭐⭐ AND THE SETTINGS, HERE, IN C, BEFORE THE LIST EXISTS. ⚖ THE CONSTRAINT IS ORDERING AND NOT
         * OWNERSHIP: *"i dont mind it can be the boot's c as well… as long as it happens before the other
         * opcodes. i said sys because the system register needs to be instantiated."*
         * ⇒ This is the only place that satisfies both ends. It runs AFTER the register exists — the hatch
         * above was just published through it — and BEFORE any init is evaluated; and it is HANDED THE TEXT
         * AS AN ARGUMENT, which is the one thing an init can never be, because a call with no arguments
         * cannot be given a string.
         * ⛳ NO CONFIG FILE STILL WRITES THE ROW, EMPTY, so an init has ONE failure mode — the key is not
         * there — instead of two. ⛳ AND THE FILE'S STRING IS RELEASED THE MOMENT THE PARSE ANSWERS: the keys
         * became hashes and the values their own strings, so nothing keeps a pointer into the text. */
        if (ok) {
            const uint64_t text = sys__string__create(settings_here, settings_length);
            ok = (text != 0ull);
            if (ok) {
                const uint64_t settings = sys__settings__from_string(text);
                (void)sys__heap_object__release(text);
                if (settings != 0ull) {
                    const sys__heap_node held = sys__heap_object__reference_to(settings);
                    ok = sys__node_array__set(all_settings, w, &held);
                    (void)sys__heap_object__release(settings);   /* the array of them holds it now, or nothing does */
                } else ok = false;
            }
        }

        const uint64_t list = ok ? sys__list__create() : 0ull;
        ok = ok && (list != 0ull);
        PACKAGE_LIST(ENG__ABI__INIT_GATHER)

        uint32_t turns = 0u;
        while (ok && sys__sublist__length(list) > 0ull) {
            if (++turns > ENG__ABI__PACKAGE_INIT_TURNS) { ok = false; break; }
            /* The head, and the form that calls it. Two cells: the verb the driver is about to run, and the
             * list it is being handed so it can ask what is still pending. */
            sys__heap_node cells[2];
            cells[0] = sys__sublist__nth(list, 0ull);
            /* ⛔⛔ QUOTED, AND IT IS NOT A STYLE CHOICE. The scan descends into any OBJECT_REFERENCE that
             * names a list and reduces it before applying the form around it — so handing the init list over
             * that way makes the evaluator EVALUATE THE LIST OF INITS as if it were a form, applying the
             * first call with the wrong arity. A quote is what says "this is the argument, not the next
             * thing to run", and it still carries its hold like any other reference. */
            cells[1].dtype = SYS__KIND__QUOTED_LIST; cells[1].num_args = 0u;
            cells[1].op_code = 0ull; cells[1].args[0] = list;
            const uint64_t form = sys__list__create_executable(cells, 2ull);
            if (form == 0ull) { ok = false; break; }
            /* ⛳ THE HEAD IS DROPPED BEFORE THE CALL RUNS, NOT AFTER, and the order is the correctness: a
             * deferral appends a copy of itself, so dropping afterwards would take the copy off again when
             * the list holds exactly one call. */
            (void)sys__sublist__discard(list, 0ull, 1ull);
            const sys__heap_node answer = eng__eval(0ull, form, here);
            /* ⛔⛔ A REFUSING INIT ANSWERS AN ERROR OBJECT, SO THAT IS THE SHAPE ASKED FOR. An init refuses the
             * way every verb does — `sys__package__refuses` leaves the form holding a REFERENCE to an error it
             * made — while an inline `VALUE_ERROR` is minted by the door that refuses a base before any program
             * runs (`eng__abi__zzprivate_run`), which this driver does not pass through. Both shapes are asked,
             * because both are what "an error" is in this tree; the object is the one a refusal here takes.
             * `MEASURED`, `nn`'s init made to refuse: a test of the inline shape alone let the boot
             * answer OK with the refusal's fault raised beside it, and the host harness reads the answer as an
             * `OBJECT_REFERENCE` naming an error whose code is the refusal's.
             * ⛳ THE ANSWER IS BORROWED FROM THE FORM, so it is read here, before the release below lets it go. */
            const bool refused = answer.dtype == SYS__KIND__VALUE_ERROR
                              || (answer.dtype == SYS__KIND__OBJECT_REFERENCE && sys__error__is(answer.args[0]));
            if (refused) ok = false;
            (void)sys__heap_object__release(form);
        }

        if (list != 0ull) (void)sys__heap_object__release(list);
    }   /* the worker */
    sys__silicon__host_act_as(-1);
    if (eng__abi__zzpackage_workers[0].family != SYS__SILICON_FAMILY__NONE)
        (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[0].family, eng__abi__zzpackage_workers[0].device);
    if (!ok)
        (void)sys__silicon__cas_u32((unsigned int*)status, (unsigned int)ENG__BOOT__OK,
                                    (unsigned int)ENG__BOOT__PACKAGES_REFUSED);
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    const uint64_t*      slices;
    const unsigned char* settings_text;
    const uint64_t*      settings_lengths;
    int*                 status;
} eng__abi__k_stand_up_packages__args;
static void eng__abi__k_stand_up_packages__run(const void* at) {
    const eng__abi__k_stand_up_packages__args* a = (const eng__abi__k_stand_up_packages__args*)at;
    eng__abi__k_stand_up_packages(a->slices, a->settings_text, a->settings_lengths, a->status);
}
#undef ENG__ABI__INIT_GATHER
#undef ENG__ABI__INIT_ROW
#undef ENG__ABI__INIT_CALL
#undef ENG__ABI__ROOM_ROW

/* ⭐⭐ ONE PIECE OF WORK ON THE MACHINE, FROM THE DOOR THAT ASKS FOR IT. The machine is this process, so a door
 * that needs one runner runs the work itself — reading the caller's cells and bytes where they are and
 * answering into its own locals — instead of copying them into the machine's memory, launching a kernel
 * and reading an answer back out of a slot: that round trip was the shape of a machine on a card.
 * ⛳ WHAT IT STILL OWES IS WHAT EVERY RUNNER OWES: it is runner zero, the block its allocations are carved
 *   for (`ENG__LAUNCH_HERE`), and the carve is armed before the work and its watermark flushed after —
 *   the pairing `arm_then_flush` holds every runner to, checked here on this macro and on every use of it.
 * ⛔ SO THE WORK MUST NOT `return`: that would leave the door with the carve still armed. */
#define ENG__ABI__ON_THE_MACHINE(...)                                                                  \
    ENG__LAUNCH_HERE(sys__heap__forget_base(); __VA_ARGS__; sys__heap__stop_carving())

/* The answer a launch of several runners leaves in the machine's answer slot, once the work is done. */
static inline bool eng__abi__zzpackage_collect_answer(void* into, size_t bytes) {
    return sys__gpu__compute_completed(eng__abi__zzpackage_machine.family) &&
           sys__gpu__memory_read(eng__abi__zzpackage_machine.family, into, eng__abi__zzpackage_answer, bytes);
}

/* ⭐ A WORKER'S OWN FILE: when the config has a `worker<w>_params{…}` section, the file its packages' host inits
 * read is `device_params{…}` holding that section — so a package sizes a worker's rooms from the worker's own
 * figures without knowing there are workers. Null for worker 0 and for one with no section of its own: it
 * reads the config as it is. The caller frees it. */
static char* eng__abi__zzprivate_worker_config(const char* config, uint64_t length, unsigned int w) {
    if (w == 0u || config == 0 || length == 0ull) return 0;
    char name[32];
    snprintf(name, sizeof name, "worker%u_params", w);
    const char* body = 0;
    uint64_t body_length = 0ull;
    if (sys__settings__host_section(config, length, name, &body, &body_length) != SYS__SETTINGS__READ_FOUND) return 0;
    char* own = (char*)malloc((size_t)body_length + 32u);
    if (own == 0) return 0;
    memcpy(own, "device_params{\n", 15u);
    memcpy(own + 15u, body, (size_t)body_length);
    memcpy(own + 15u + body_length, "\n}\n", 4u);
    own[19u + body_length] = '\0';
    return own;
}

/* The workers' sections into the stand-up's text, one after another. */
static bool eng__abi__zzprivate_write_sections(unsigned char* text, const char* const* sections, const uint64_t* lengths,
                                               unsigned int workers) {
    uint64_t at = 0ull;
    for (unsigned int w = 0u; w < workers; ++w) {
        if (lengths[w] != 0ull && !sys__gpu__memory_write(eng__abi__zzpackage_machine.family, text + at, sections[w], (size_t)lengths[w]))
            return false;
        at += lengths[w];
    }
    return true;
}

extern "C" {

/* Named here because the boot calls it: a boot whose packages refuse takes the machine back down rather
 * than handing over one that is up by every check and missing what its packages were going to build. */
void eng_abi_shutdown(void);

/* Stand the engine up. Every figure the caller may leave at zero for the boot to settle, and the answer
 * is why not, rather than merely no.
 *
 * ⭐ `config` IS THE WHOLE FILE — `boot_commands{…}` and `device_params{…}`, in the `key:value` lines the
 * machine reads. Null or empty is a complete answer: every package uses its own default and the
 * settings row holds an empty sealed dictionary, so an init has ONE failure mode rather than two.
 * ⛳ THE `device_params` SECTION IS SLICED RATHER THAN RENDERED — `sys__settings__from_string` is
 * handed the bytes a human wrote, so there is no serialiser anywhere that could disagree with a parser.
 * ⚖ THE ORDER, RULED: *"the boot setup does the various memory allocations and then starts the boot
 * process."* Each package's host init runs FIRST, before `eng__boot`, because what it takes is the room
 * the machine is about to be told about. */
int eng_abi_boot_with(unsigned int blocks, unsigned int chunks, unsigned int chunk_nodes,
                      const char* config, unsigned long long config_bytes) {
    if (eng__abi__zzpackage_up()) return ENG__BOOT__OK;   /* already up: asking twice is not an error */

    const uint64_t config_length = (config == 0) ? 0ull : (uint64_t)config_bytes;

    /* ── ⓪ THE SILICON, BEFORE ANYTHING IS TAKEN ─────────────────────────────────────────────────────
     * The machine itself is always the host's; what is chosen is where the worker and the package rooms
     * go. Every room below names it, so it is settled first: the family chosen by name, or the first one
     * offered — the host's. A family not offered refuses the boot here, as itself, rather than surfacing
     * later as a device with no memory. */
    const sys__silicon_family__id chosen = (eng__abi__zzpackage_silicon_family_wanted != SYS__SILICON_FAMILY__NONE)
                                    ? eng__abi__zzpackage_silicon_family_wanted : sys__silicon_family__first();
    const sys__silicon_family__id family = SILICON_FAMILIES__HOST;     /* the machine: where the evaluator runs */
    if (sys__silicon_family__of(family) == 0 || sys__silicon_family__of(chosen) == 0) return ENG__BOOT__NO_SILICON;
    /* ⭐ THE WORKERS: the ones chosen, or one — the chosen family's device 0. */
    const unsigned int workers = eng__abi__zzpackage_wanted_count != 0u ? eng__abi__zzpackage_wanted_count : 1u;
    for (unsigned int w = 0u; w < SYS__SILICON__WORKERS_MAX; ++w) {
        eng__abi__zzpackage_workers[w].family = SYS__SILICON_FAMILY__NONE;
        eng__abi__zzpackage_workers[w].device = 0u;
        eng__abi__zzpackage_workers[w].fault_word = 0;
        eng__abi__zzpackage_workers[w].heap_card = 0;
        eng__abi__zzprivate_rooms_taken[w] = 0u;
    }
    for (unsigned int w = 0u; w < workers; ++w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        me->family = eng__abi__zzpackage_wanted_count != 0u ? eng__abi__zzpackage_wanted_family[w] : chosen;
        me->device = eng__abi__zzpackage_wanted_count != 0u ? eng__abi__zzpackage_wanted_device[w] : 0u;
        if (sys__silicon_family__of(me->family) == 0) return ENG__BOOT__NO_SILICON;
        for (uint64_t id = 0ull; id < (uint64_t)package_count; ++id) eng__abi__zzprivate_rooms[w][id].family = me->family;
    }
    eng__abi__zzpackage_worker_count = workers;
    sys__silicon__host_set_worker_count(workers);
    eng__launch__forget_workers();
    for (unsigned int w = 0u; w < workers; ++w)
        eng__launch__set_worker(w, eng__abi__zzpackage_workers[w].family, eng__abi__zzpackage_workers[w].device);
    eng__abi__zzpackage_worker_silicon_family = eng__abi__zzpackage_workers[0].family;

    /* ── ① THE HOST INITS, A WORKER AT A TIME AND IN ROSTER ORDER, EACH TAKING ITS OWN ─────────────────
     * ⛳ THE WORKER TAKES ITS DEVICE BEFORE ANYTHING IS ALLOCATED ON IT, because an allocation lands on
     *   whichever device is current. On the host device 0 is the process itself and binding it always
     *   succeeds; a loaded family with no such device refuses here, as the silicon.
     * ⛳ A WORKER'S SETTINGS ARE ITS OWN SECTION, `worker<w>_params{…}`, or `device_params{…}` when it has none
     *   — handed to its packages' host inits as a file whose `device_params` is that section.
     * ⛔ AND THE UNWIND IS WRITTEN WITH THEM RATHER THAN AFTER THEM: a loop turns ONE failure point into
     * N, so if package k refuses, packages 0..k-1 are already holding memory in a process that keeps
     * running. `rooms_taken` counts only the ones that answered true, and the give-back walks it back. */
    bool ok = true;
    for (unsigned int w = 0u; ok && w < workers; ++w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        if (!sys__silicon__bind_device(me->family, me->device)) { eng__abi__zzprivate_give_rooms_back(); return ENG__BOOT__NO_SILICON; }
        char* own = eng__abi__zzprivate_worker_config(config, config_length, w);
        const char* wconfig = own != 0 ? own : config;
        const uint64_t wconfig_length = own != 0 ? (uint64_t)strlen(own) : config_length;
        PACKAGE_LIST(ENG__ABI__HOST_INIT_ROW)
        free(own);
        /* The worker's fault word, on its own device: what every door accumulates an overflow into. */
        if (ok && (!sys__gpu__memory_allocate(me->family, (void**)&me->fault_word, sizeof(uint32_t))
                || !sys__gpu__memory_zerofill(me->family, me->fault_word, sizeof(uint32_t)))) ok = false;
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED
        if (ok) sys__engine__ctx_bind(w, me->family, me->device, me->fault_word);
#endif
    }
    if (!ok) {
        eng__abi__zzprivate_give_rooms_back();   /* gives the words back too, where they were taken */
        return ENG__BOOT__NO_MEMORY;
    }
    (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[0].family, eng__abi__zzpackage_workers[0].device);

    /* ⭐ THE HEAP'S SIZE IS A `boot_commands` FIGURE TOO — `sys__alloc_size_mb:128` — and it is read HERE
     * rather than by `sys`'s host init, which is the one asymmetry in the loop and has a reason. The heap
     * is the allocation that exists BEFORE any package does: `eng__boot` takes it, the register and the
     * blocks are carved out of it, and a package's host init runs before all of that. So `sys` cannot
     * take its own the way `nn` does — its memory IS the heap, and the engine takes it on sys's behalf.
     * ⛳ THE ARITHMETIC IS EXACT AND THAT IS WHY THE UNIT IS MEBIBYTES: a chunk is
     * `SYS__HEAP__CHUNK_NODES` × 64 bytes = 32 KiB exactly, so 128 MiB is 4,096 chunks and nothing rounds.
     * ⛔ AN EXPLICIT `chunks` ARGUMENT STILL WINS, because it is the more specific request — a caller that
     * named chunks asked for chunks, and a config is the answer for a caller that did not. */
    uint64_t heap_bytes = 0ull;
    const int heap_said = sys__settings__host_size(config, config_length, "boot_commands",
                                                   "sys__alloc_size_mb", &heap_bytes);
    if (heap_said == SYS__SETTINGS__READ_MALFORMED) {
        eng__abi__zzprivate_give_rooms_back();
        return ENG__BOOT__NO_MEMORY;
    }
    unsigned int want_chunks = chunks;
    if (heap_said == SYS__SETTINGS__READ_FOUND && chunks == 0u) {
        const uint64_t per_chunk = (uint64_t)SYS__HEAP__CHUNK_NODES * (uint64_t)sizeof(sys__heap_node);
        const uint64_t asked     = heap_bytes / per_chunk;
        if (asked == 0ull || asked > 0xFFFFFFFFull) {
            eng__abi__zzprivate_give_rooms_back();
            return ENG__BOOT__NO_MEMORY;
        }
        want_chunks = (unsigned int)asked;
    }

    EngineBootRequest want;
    /* ⛳ A BLOCK A WORKER AT LEAST: a program puts work on worker `w` by computing onto block `w`. */
    want.blocks = (workers > 1u && blocks < workers) ? workers : blocks; want.chunks = want_chunks; want.chunk_nodes = chunk_nodes; want.family = family;
    if (!eng__boot(&want, &eng__abi__zzpackage_machine)) {
        eng__abi__zzprivate_give_rooms_back();
        return eng__abi__zzpackage_machine.status;
    }
    /* ⛳ THE CHANNEL IS PART OF BEING UP, SO A BOOT THAT CANNOT MAKE ONE REFUSED. Leaving it optional
     * would mean a machine that is up by every check and cannot be looked at while it works, and the
     * caller would find out at the one moment there is nothing to be done about it.
     * On the host both views of the landing room are the same bytes, so the second is not kept. */
    void* landing_card_view = 0;
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family,
                                 (void**)&eng__abi__zzpackage_answer, sizeof(sys__heap_node))  ||
        !sys__gpu__side_open(eng__abi__zzpackage_machine.family,
                                      &eng__abi__zzpackage_channel)                            ||
        !sys__gpu__memory_allocate_host_ram_mapped(eng__abi__zzpackage_machine.family,
                                (void**)&eng__abi__zzpackage_landing, &landing_card_view,
                                sizeof(sys__heap_node))) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_answer);
        sys__gpu__side_close(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_channel);
        sys__gpu__memory_free_host_ram(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_landing);
        eng__boot_release(&eng__abi__zzpackage_machine);
        eng__abi__zzpackage_answer  = 0;
        eng__abi__zzpackage_channel = 0;
        eng__abi__zzpackage_landing = 0;
        eng__abi__zzprivate_give_rooms_back();   /* the packages' rooms were taken above, and go back too */
        return ENG__BOOT__NO_MEMORY;
    }

    /* ⭐ THE HEAP, WRITABLE BY THE WORKER'S CARD — so a kernel can put a value straight into a node a program
     *   made for it (`nn__vector__top_k`'s answer). A family that cannot register RAM it did not allocate
     *   answers no, and the machine is up all the same: the verb copies instead. */
    /* ⛳ EACH WORKER REGISTERS IT ON ITS OWN DEVICE; one that cannot — a second card of a family that already
     *   holds the range — keeps 0 and its verbs copy. */
    for (unsigned int w = 0u; w < workers; ++w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        me->heap_card = 0;
        if (!sys__silicon__bind_device(me->family, me->device)
         || !sys__gpu__memory_register_host(me->family, eng__abi__zzpackage_machine.heap,
                                            (size_t)eng__abi__zzpackage_machine.heap_bytes, &me->heap_card))
            me->heap_card = 0;
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED
        sys__engine__ctx_heap(w, eng__abi__zzpackage_machine.heap, (sys__heap_node*)me->heap_card,
                              eng__abi__zzpackage_machine.heap_bytes / sizeof(sys__heap_node));
#endif
    }
    (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[0].family, eng__abi__zzpackage_workers[0].device);

    /* ── AND THE PACKAGES LAST, BECAUSE THEY RUN AS A PROGRAM ────────────────────────────────────────
     * ⛳ EVERYTHING ABOVE HAD TO HAPPEN FIRST AND NOT BY PREFERENCE: this evaluates a list, so it needs a
     * heap, a register, a started block with a computing base and an evaluator standing on all three.
     * That is exactly the state a package's init can therefore assume, which is what makes the ordering
     * question between packages theirs rather than the boot's. */
    int  packaged = ENG__BOOT__OK;
    int* reported = 0;
    /* What the stand-up is handed as arguments, because runner zero writes both into the machine before
     * the first init runs: the slices each host init took, and the settings text. */
    uint64_t* slices   = 0;
    unsigned char* text = 0;
    uint64_t here_slices[SYS__SILICON__WORKERS_MAX * (size_t)package_count * SYS__PACKAGE__ROOM_SLICE];
    for (uint64_t w = 0ull; w < (uint64_t)workers; ++w)
        for (uint64_t id = 0ull; id < (uint64_t)package_count; ++id) {
            const sys__package_room* r = &eng__abi__zzprivate_rooms[w][id];
            uint64_t* slice = &here_slices[(w * (uint64_t)package_count + id) * SYS__PACKAGE__ROOM_SLICE];
            slice[0] = (uint64_t)(uintptr_t)r->at;
            slice[1] = r->bytes;
            slice[2] = (uint64_t)(uintptr_t)r->ram_at;
            slice[3] = (uint64_t)(uintptr_t)r->ram_card_at;
            slice[4] = r->ram_bytes;
        }
    /* ⛳ EACH WORKER'S SECTION, SLICED RATHER THAN RENDERED — `worker<w>_params`, or `device_params` — one after
     * another, each its own length. A config with neither hands over nothing, which the parse answers with an
     * empty sealed dictionary — the no-config case. */
    const char* sections[SYS__SILICON__WORKERS_MAX];
    uint64_t    lengths[SYS__SILICON__WORKERS_MAX];
    uint64_t    device_length = 0ull;
    for (unsigned int w = 0u; w < workers; ++w) {
        sections[w] = 0; lengths[w] = 0ull;
        if (config_length == 0ull) continue;
        char own[32];
        snprintf(own, sizeof own, "worker%u_params", w);
        int found = w == 0u ? SYS__SETTINGS__READ_ABSENT
                            : sys__settings__host_section(config, config_length, own, &sections[w], &lengths[w]);
        if (found == SYS__SETTINGS__READ_ABSENT)
            found = sys__settings__host_section(config, config_length, "device_params", &sections[w], &lengths[w]);
        if (found == SYS__SETTINGS__READ_MALFORMED) {
            /* ⛔ A SECTION THAT OPENS AND NEVER CLOSES IS A FILE CUT SHORT, AND IT IS REFUSED RATHER THAN
             * READ AS ABSENT. Reading it as absent is what this did until a test caught it, and the cost
             * was the worst kind: every package quietly fell back to its default and the machine booted
             * at a size nobody asked for, with nothing anywhere saying so.
             * ⛳ AND THE MACHINE COMES DOWN WITH THE ROOMS: the heap, the answer slot, the channel and the
             *   landing are all up by now, and a machine left standing without its rooms would answer the
             *   next boot "already up". */
            eng__abi__zzprivate_give_rooms_back();
            eng_abi_shutdown();
            return ENG__BOOT__NO_MEMORY;
        }
        if (found != SYS__SETTINGS__READ_FOUND) { sections[w] = 0; lengths[w] = 0ull; }
        device_length += lengths[w];
    }
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&reported, sizeof(int)) ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, reported, sizeof(int))) {
        packaged = ENG__BOOT__NO_MEMORY;
    } else if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&slices, sizeof here_slices) ||
               !sys__gpu__memory_write(eng__abi__zzpackage_machine.family, slices, here_slices, sizeof here_slices)) {
        packaged = ENG__BOOT__NO_MEMORY;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, slices); slices = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, reported);
    } else if (device_length != 0ull &&
               (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&text, (size_t)device_length) ||
                !eng__abi__zzprivate_write_sections(text, sections, lengths, workers))) {
        packaged = ENG__BOOT__NO_MEMORY;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, text); text = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, slices); slices = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, reported);
    } else if (!eng__abi__zzpackage_free_to_launch()) {
        /* ⛳ NOTHING CAN BE RESIDENT HERE — this call is what built the machine — so the check is an
         * assertion rather than a wait. It is written anyway because every door that settles owes it, and
         * a door exempt by argument is a door the next reader has to re-derive the argument for. */
        packaged = ENG__BOOT__LAUNCH_FAILED;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, text); text = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, slices); slices = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, reported);
    } else {
        const eng__abi__k_stand_up_packages__args args = { slices, text, lengths, reported };
        ENG__LAUNCH_MANY(eng__abi__k_stand_up_packages__run, eng__abi__zzpackage_machine.blocks, &args);
        if (!sys__gpu__compute_completed(eng__abi__zzpackage_machine.family) ||
            !sys__gpu__memory_read(eng__abi__zzpackage_machine.family, &packaged, reported, sizeof(int))) {
            packaged = ENG__BOOT__LAUNCH_FAILED;
        }
        /* ⛳ BOTH ARE THE CROSSING'S AND NOT THE MACHINE'S: the stand-up copied the slices into the hatch
         * and the text into a string, so neither is wanted once the launch has settled. */
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, text); text = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, slices); slices = 0;
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, reported);
    }
    if (packaged != ENG__BOOT__OK) {
        eng__abi__zzprivate_give_rooms_back();
        /* A machine whose packages did not stand up is not a machine to hand back. It comes down the
         * same way a failed boot does, so a caller is never given one that is up by every check and
         * missing whatever its packages were going to build. */
        eng_abi_shutdown();
        return packaged;
    }
    return ENG__BOOT__OK;
}

/* The door without a config, which is every caller that had one before this and a great many tests.
 * ⛳ IT IS NOT A DEFAULT ARGUMENT: this is an `extern "C"` surface, so a second entry point is what a
 * defaulted parameter would have to become anyway, and writing it out means the C ABI never changes
 * shape under a caller that was compiled against it. */
int eng_abi_boot(unsigned int blocks, unsigned int chunks, unsigned int chunk_nodes) {
    return eng_abi_boot_with(blocks, chunks, chunk_nodes, 0, 0ull);
}

void eng_abi_shutdown(void) {
    /* ⛔⛔ FIRST, AND EVERYTHING BELOW IS CONDITIONAL ON IT. Everything below hands the pool back, and a
     * resident launch is a runner per block reading out of it — so a shutdown that went ahead anyway
     * would not leak, it would leave runners walking memory nobody owns.
     * ⛳ SO A FAILED STAND-DOWN RETURNS AND DOES NOTHING ELSE, which LEAKS, deliberately and in the safe
     * direction: the launch is still out there, the memory it reads is still its own, and a caller may
     * ask again. Leaking until the process ends costs nothing, because ending the process reclaims it. */
    if (!eng__abi__zzpackage_stand_down()) return;
    sys__silicon__host_act_as(-1);                 /* the caller is nobody's worker once there are none */
    sys__gpu__side_close(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_channel);
    eng__abi__zzpackage_channel = 0;
    sys__gpu__memory_free_host_ram(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_landing);
    /* ⭐ EACH PACKAGE GIVES BACK WHAT IT TOOK, in reverse. A dead process' memory comes back on its own,
     * so the crash path needs nothing — but a shutdown is not a death, and a caller that boots and shuts
     * down in a loop keeps whatever this does not give back. `MEASURED` with the heap on a
     * card: eleven boot/shutdown cycles stayed flat because this ran, where without it each would have
     * kept the engine's footprint. ▶ `measurements/2026-09-18_device_memory_on_process_death.md`. */
    eng__abi__zzprivate_give_rooms_back();
    eng__abi__zzpackage_landing = 0;
    sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_answer);
    eng__abi__zzpackage_answer = 0;
    /* ⛳ EACH WORKER THAT REGISTERED THE HEAP UNREGISTERS IT, ON ITS OWN DEVICE */
    for (unsigned int w = 0u; w < eng__abi__zzpackage_worker_count; ++w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        if (me->heap_card != 0 && sys__silicon__bind_device(me->family, me->device))
            sys__gpu__memory_unregister_host(me->family, eng__abi__zzpackage_machine.heap);
        me->heap_card = 0;
    }
    if (eng__abi__zzpackage_worker_count != 0u)
        (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[0].family, eng__abi__zzpackage_workers[0].device);
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED
    for (unsigned int w = 0u; w < SYS__SILICON__WORKERS_MAX; ++w) sys__engine__ctx_heap(w, 0, 0, 0ull);
#endif
    eng__boot_release(&eng__abi__zzpackage_machine);
}

const char* eng_abi_status_text(int status) { return eng__boot__why(status); }

/* What the boot settled on, so a caller can report the machine it actually got rather than the one it
 * asked for. Zero BLOCKS when nothing is up, and a machine that is up has at least one, so that is the
 * field the question is asked of.
 * ⛔ `chunk_nodes` IS NOT PART OF THAT ANSWER AND MUST NOT BE READ AS ONE. It is the compiled width,
 * which `eng__boot_release` leaves standing while it zeroes the two figures the boot settled — so a
 * caller testing all three for zero reads a live-looking width off a machine that is down. */
void eng_abi_geometry(unsigned int* blocks, unsigned int* chunks, unsigned int* chunk_nodes) {
    if (blocks)      *blocks      = eng__abi__zzpackage_machine.blocks;
    if (chunks)      *chunks      = eng__abi__zzpackage_machine.chunks;
    if (chunk_nodes) *chunk_nodes = eng__abi__zzpackage_machine.chunk_nodes;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_MACHINE_CUH */
