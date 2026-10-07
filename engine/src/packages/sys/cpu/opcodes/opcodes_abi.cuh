#ifndef SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI_CUH
#define SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI_CUH
/* ══ `sys` VERBS IN THE RULED SHAPE — the bodies. ▶ `verb_abi.cuh` for the bridge and who owns what. ══
 */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/defaults.cuh"      /* the dials the verbs read */
#include "../heap_node__header.cuh"          /* the node every value in this language is made of */
#include "../heap__header.cuh"               /* where a verb's room comes from */
#include "../heap_object__header.cuh"        /* holds, types, clone and create */
#include "../bindings__header.cuh"           /* names and their scopes */
#include "../computing_base__header.cuh"     /* compute, completed and result */
#include "../dictionary.cuh"                 /* put and get */
#include "../list__header.cuh"               /* the lists a verb is handed and answers */
#include "../node_array__header.cuh"         /* the picture a program travels as */
#include "../package__header.cuh"            /* package init */
#include "../procedure.cuh"                  /* defun */
#include "../string.cuh"                     /* a path's bytes, for file open */
#include "../../contracts/objects/file.cuh"  /* file open's fault word */
#include "../symbol.cuh"                     /* names by number */
#include "../system_register__header.cuh"    /* the register verbs */
#include "../silicon/silicon__header.cuh"    /* the block a verb runs on */
#include "opcodes.cuh"                       /* truth, and the fault words every verb shares */
#include "verb_abi.cuh"                      /* the bridge these bodies are called through */
#include "opcodes_abi__header.cuh"           /* their declarations */

/* `(sys__eq a b)` — the same kind and the same word. */
static __device__ sys__heap_node sys__opcodes__zzabi_eq(const sys__heap_node* argv, unsigned argc,
                                                        sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return sys__opcodes__truth(argv[0].dtype == argv[1].dtype && argv[0].args[0] == argv[1].args[0]);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_eq, sys__opcodes__zzabi_eq)

/* ══ ⭐⭐ THE ARITHMETIC — TWO BODIES, NINE VERBS ══════════════════════════════════════════════════════
 *
 * Six typed verbs choose their domain by name; three dispatchers choose it from the first operand's kind.
 * All nine share the two bodies below, so the arithmetic has one definition.
 *
 * ⛳ `sys__opcodes__zzpackage_as_real` IS SHARED AS IT STANDS: it already takes a node rather than a
 * form. `zzprivate_operands3` reads a form, and its one caller is the `SILVANN_ONE_READ` arm of
 * `sys__int__add` in `opcodes.cuh` — which is also why that verb's contract row names a macro
 * (`SYS__OPCODES__ZZABI_INT_ADD_APPLY`, in `language_contract.cuh`) rather than its adapter.
 */

/* Two operands, for all nine. Local to this section; undefined after the last of them. */
#define SYS__OPCODES__ZZABI_ARITY2 \
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);

/* The two bodies the nine verbs share. ⛳ Each answers `false` for operands outside its domain and fills
 * `out` otherwise; the verb turns that into an error value or an answer. `which` picks the operation:
 * 0 adds, 1 subtracts, 2 compares. */
static __device__ inline bool sys__opcodes__zzabi_int_pair(const sys__heap_node* argv, int which,
                                                            sys__heap_node* out) {
    const sys__heap_node a = argv[0];
    const sys__heap_node b = argv[1];
    /* ⛔ AN INT NARROWS NOTHING: its partner must be an int too. ⚖ *"an int adding a float won't."* */
    if (a.dtype != SYS__KIND__VALUE_INT || b.dtype != SYS__KIND__VALUE_INT) return false;
    /* ⚖ AN INT COMPARES SIGNED: `(sys__int__sub 2 5)` wraps to the bits of -3, and a comparison that
     * read them unsigned would call it larger than every positive number. Add and subtract are the
     * same bits either way, so only the comparison has to say which it means. */
    if (which == 2) { *out = sys__opcodes__truth((int64_t)a.args[0] < (int64_t)b.args[0]); return true; }
    out->dtype = SYS__KIND__VALUE_INT; out->num_args = 0u; out->op_code = 0ull;
    out->args[0] = (which == 0) ? (a.args[0] + b.args[0]) : (a.args[0] - b.args[0]);
    return true;
}

static __device__ inline bool sys__opcodes__zzabi_real_pair(const sys__heap_node* argv, int which,
                                                             sys__heap_node* out) {
    const sys__heap_node a = argv[0];
    const sys__heap_node b = argv[1];
    double av = 0.0, bv = 0.0;
    /* ⭐ THE FIRST OPERAND MUST BE A FLOAT — it is what chose this domain — while the SECOND may be
     * either, and an int widens into it. */
    if (a.dtype != SYS__KIND__VALUE_FLOAT || !sys__opcodes__zzpackage_as_real(&a, &av)
        || !sys__opcodes__zzpackage_as_real(&b, &bv)) return false;
    if (which == 2) { *out = sys__opcodes__truth(av < bv); return true; }
    out->dtype = SYS__KIND__VALUE_FLOAT; out->num_args = 0u; out->op_code = 0ull;
    out->args[0] = sys__heap_node__real_bits((which == 0) ? (av + bv) : (av - bv));
    return true;
}

/* ── THE SIX TYPED VERBS ─────────────────────────────────────────────────────────────────────────────
 * ⛳ WRITTEN OUT RATHER THAN PASTED BY A MACRO, so each verb's name is in the text where a tool reading
 * the source can find it. The bodies are shared; only the six signatures repeat. */

static __device__ sys__heap_node sys__opcodes__zzabi_int_add(const sys__heap_node* argv, unsigned argc,
                                                              sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_int_pair(argv, 0, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_float_add(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_real_pair(argv, 0, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_int_sub(const sys__heap_node* argv, unsigned argc,
                                                              sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_int_pair(argv, 1, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_float_sub(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_real_pair(argv, 1, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_int_less(const sys__heap_node* argv, unsigned argc,
                                                               sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_int_pair(argv, 2, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_float_less(const sys__heap_node* argv, unsigned argc,
                                                                 sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    if (!sys__opcodes__zzabi_real_pair(argv, 2, &out)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

/* ── THE THREE DISPATCHERS ───────────────────────────────────────────────────────────────────────────
 * ⚖ *"add will take the first element, check its type and run the add for that type."* They choose a
 * domain and then answer; every int program ever frozen reduces exactly as it did before. */

static __device__ sys__heap_node sys__opcodes__zzabi_add(const sys__heap_node* argv, unsigned argc,
                                                          sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    bool mine = false;
    if (argv[0].dtype == SYS__KIND__VALUE_FLOAT)    mine = sys__opcodes__zzabi_real_pair(argv, 0, &out);
    else if (argv[0].dtype == SYS__KIND__VALUE_INT) mine = sys__opcodes__zzabi_int_pair(argv, 0, &out);
    if (!mine) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_sub(const sys__heap_node* argv, unsigned argc,
                                                          sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    bool mine = false;
    if (argv[0].dtype == SYS__KIND__VALUE_FLOAT)    mine = sys__opcodes__zzabi_real_pair(argv, 1, &out);
    else if (argv[0].dtype == SYS__KIND__VALUE_INT) mine = sys__opcodes__zzabi_int_pair(argv, 1, &out);
    if (!mine) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

static __device__ sys__heap_node sys__opcodes__zzabi_less(const sys__heap_node* argv, unsigned argc,
                                                           sys__engine__ctx* ctx) {
    (void)ctx; SYS__OPCODES__ZZABI_ARITY2
    sys__heap_node out;
    bool mine = false;
    if (argv[0].dtype == SYS__KIND__VALUE_FLOAT)    mine = sys__opcodes__zzabi_real_pair(argv, 2, &out);
    else if (argv[0].dtype == SYS__KIND__VALUE_INT) mine = sys__opcodes__zzabi_int_pair(argv, 2, &out);
    if (!mine) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return out;
}

#undef SYS__OPCODES__ZZABI_ARITY2

SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_int_add,    sys__opcodes__zzabi_int_add)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_float_add,  sys__opcodes__zzabi_float_add)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_int_sub,    sys__opcodes__zzabi_int_sub)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_float_sub,  sys__opcodes__zzabi_float_sub)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_int_less,   sys__opcodes__zzabi_int_less)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_float_less, sys__opcodes__zzabi_float_less)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_add,        sys__opcodes__zzabi_add)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_sub,        sys__opcodes__zzabi_sub)
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_less,       sys__opcodes__zzabi_less)

/* ══ ⭐⭐ THE OBJECT VERBS AND THE REGISTER'S TWO DOORS ═══════════════════════════════════════════════
 *
 * ⛔ EVERY VERB STATES ITS ARITY BEFORE IT READS AN OPERAND. `argv` holds exactly `argc` of them, and
 * `argv[0]` with `argc == 0` is uninitialised stack — not a refusal, an answer or a fault, just whatever
 * was there. A short call answers an ARITY error, and an error is a value the program can see.
 */

/* `(sys__clone x)` — one of the two verbs here that needs an arena. ▶ `sys__engine__ctx.base`.
 * ⛳ THE COPY IS CARVED FROM THE ASKING BLOCK'S OWN CHUNK, so one block's liveness never depends on
 * another's. */
static __device__ sys__heap_node sys__opcodes__zzabi_clone(const sys__heap_node* argv, unsigned argc,
                                                            sys__engine__ctx* ctx) {
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    const uint64_t copy = sys__heap_object__clone(argv[0].args[0], ctx->base);
    if (copy == 0ull) return sys__heap_node__nothing();   /* the clone already raised */
    /* ⚖⚖ A COPY IS A DATUM, NOT A FORM. As a plain object reference it would name a list, the scan in
     * `eng__eval__zzprivate_first_form` would descend into it, and the evaluator would try to apply its
     * first element as a verb. Something you have just been handed a copy of is data, and the tag says
     * so. `create` answers the same way for the same reason.
     * ⛳ The maker's hold on the copy is the hold this answers with; the bridge gives it back after
     * publishing. ▶ the ownership rule in `verb_abi.cuh`. */
    sys__heap_node made = sys__heap_object__reference_to(copy);
    if (sys__heap_node__carries_reference(made.dtype)) made.dtype = SYS__KIND__QUOTED_LIST;
    return made;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_clone, sys__opcodes__zzabi_clone)

/* `(sys__type x)` — the kind, as a value a program can compare. The name to compare it against is already
 * published: a kind's row in the shared node-type list carries the spelling the host binds, so a composer
 * writes the kind and this answers with the same number. */
static __device__ sys__heap_node sys__opcodes__zzabi_type(const sys__heap_node* argv, unsigned argc,
                                                           sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    /* ⛔ ONLY A REFERENCE HAS A KIND TO READ, AND ONLY AN ADDRESSABLE ONE. Finding an object's head is
     * arithmetic on the offset, so any number answers something — a plain `5` would be read as an offset
     * and answer the kind of whatever sits there. Anything else is refused as a TYPE error. */
    if (!sys__heap_node__carries_reference(argv[0].dtype)
        || !sys__heap_object__zzpackage_addressable(argv[0].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__heap_node answer;
    answer.dtype    = SYS__KIND__VALUE_INT;
    answer.num_args = 0u;
    answer.op_code  = 0ull;
    answer.args[0]  = (uint64_t)sys__heap_object__type(
                          sys__heap_node__zzpackage_head(argv[0].args[0]) + 1);
    return answer;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_type, sys__opcodes__zzabi_type)

/* `(sys__create kind ...)` — the kind first, then up to five arguments of the kind's own. The other verb
 * here that needs an arena.
 * ⛳ THE ANSWER IS TAGGED A DATUM FOR EVERY KIND, NOT ONLY THE LISTS. Only a list is at risk from the scan,
 * but tagging just those would make what a constructor answers depend on WHICH constructor, and a caller
 * would have to know the difference to read the tag. */
static __device__ sys__heap_node sys__opcodes__zzabi_create(const sys__heap_node* argv, unsigned argc,
                                                             sys__engine__ctx* ctx) {
    if (argc < 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    /* ⛳ THE LOOP READS ONLY THE OPERANDS THAT ARE THERE; a missing one is zero. Operands past the fifth
     * are not read.
     * ⛔ A SUCCESSFUL CALL THAT RAISES IS WORSE THAN A SILENT ONE: the fault channel is single-slot, so a
     * spurious word overwrites whatever a caller was about to be told, and the one instrument that says
     * something went wrong learns to cry wolf on the ordinary path. */
    sys__heap_node rest = sys__heap_node__nothing();
    for (uint32_t i = 0u; i < 5u; ++i)
        rest.args[i] = ((unsigned)i + 1u < argc) ? argv[(unsigned)i + 1u].args[0] : 0ull;
    rest.args[5] = 0ull;
    sys__heap_node made = sys__heap_object__create(ctx->base, (sys__kind)argv[0].args[0], &rest);
    /* ⚖⚖ A CONSTRUCTOR ANSWERS A DATUM, NOT A FORM: *"create tags its answer QUOTED_LIST"*. Without
     * the tag the scan descends into what was just built and runs it. */
    if (sys__heap_node__carries_reference(made.dtype)) made.dtype = SYS__KIND__QUOTED_LIST;
    return made;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_create, sys__opcodes__zzabi_create)

/* `(sys__system_register__get row)` — ⛳ the row arrives HELD and that hold is what the verb hands over. */
static __device__ sys__heap_node sys__opcodes__zzabi_register_get(const sys__heap_node* argv,
                                                                   unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return sys__system_register__get((uint32_t)argv[0].args[0]);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_register_get, sys__opcodes__zzabi_register_get)

/* `(sys__system_register__set row value)` — ⛳ no third cell saying what kind: ⚖ *"what kind" IS A
 * PROPERTY OF THE VALUE*, and a cell repeating it would be a second place for the answer to be wrong.
 * ⛔ IT ANSWERS THE VALUE IT WROTE, AND NOTHING WHEN IT WAS REFUSED — so writing a NULL into a row and
 * being refused read the same. The fault channel tells them apart and a caller cannot. What would close it
 * is already here — `#t`/`#f` from `sys__opcodes__truth` — so the gap is a decision nobody has made, not a
 * missing piece: the value written, or whether it went in. */
static __device__ sys__heap_node sys__opcodes__zzabi_register_set(const sys__heap_node* argv,
                                                                   unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    sys__heap_node written = sys__heap_node__nothing();
    written.dtype   = argv[1].dtype;
    written.args[0] = argv[1].args[0];
    if (!sys__system_register__set((uint32_t)argv[0].args[0], written.dtype, written.args[0]))
        return sys__heap_node__nothing();      /* the setting already raised, and said which reason */
    /* The value was handed in, not made here, so the hold the answer carries is taken now. The register
     * took its own when the row was written; this one is the answer's, and the bridge gives it back. */
    if (sys__heap_node__carries_reference(written.dtype)) (void)sys__heap_object__retain(written.args[0]);
    return written;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_register_set, sys__opcodes__zzabi_register_set)

/* ══ ⭐⭐ A SCOPE PLACED BY HAND, AND THE BRANCH ══════════════════════════════════════════════════════
 *
 * ⛳ `if` ANSWERS A VALUE IT DID NOT MAKE — one of its own arms — so it takes a hold on it before
 * handing it over, and the bridge gives that hold back once the form has taken its own. `add_bindings`
 * answers a truth value, which holds nothing. `begin`, `prog1`, `remove_bindings` and `let` are the
 * evaluator's own control flow and keep the form-writing shape — the first three in `opcodes.cuh`,
 * `let` in `bindings__impl.cuh`.
 */

/* `(add_bindings env 'name value)` — a scope placed by hand. ⛳ A scope lives in the LIST rather than
 * in the evaluator's control flow, which is what lets `let` leave a matching remove behind it. */
static __device__ sys__heap_node sys__opcodes__zzabi_add_bindings(const sys__heap_node* argv,
                                                                   unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return sys__opcodes__truth(sys__bindings__add(argv[0].args[0], &argv[1], &argv[2]));
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_add_bindings, sys__opcodes__zzabi_add_bindings)

/* `(if test 'then 'else)` — ⭐⭐ IT DOES NOT EVALUATE A BRANCH. It leaves the chosen arm, unquoted,
 * where the whole form was, and the scan meets it next: a rewrite rather than a call, so nothing
 * recurses and no frame is spent.
 * ⛳ THIS IS THE VERB THAT PROVES THE TIER HAS NO SPECIAL FORMS IN THE ABI'S SENSE. Unquoting is a KIND
 * and nothing else — `eng__eval__zzprivate_first_form` descends because the answer NAMES A LIST, not
 * because `if` asked it to. ⇒ ★★ CONTROL FLOW LIVES IN THE EVALUATOR'S SCAN, NOT IN THE VERBS.
 * ⛔ THE TEST MUST BE `#t` OR `#f`. There are three outcomes — true, false, or an error, which is a
 * value — so anything else is refused as a TYPE error rather than read as true. */
static __device__ sys__heap_node sys__opcodes__zzabi_if(const sys__heap_node* argv, unsigned argc,
                                                         sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_TRUE && argv[0].dtype != SYS__KIND__VALUE_FALSE)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const sys__heap_node arm = argv[argv[0].dtype == SYS__KIND__VALUE_TRUE ? 1u : 2u];
    if (arm.dtype != SYS__KIND__QUOTED_LIST) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__heap_node chosen = arm;
    chosen.dtype = SYS__KIND__OBJECT_REFERENCE;
    (void)sys__heap_object__retain(chosen.args[0]);   /* ⛳ the hold the verb hands over */
    return chosen;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_if, sys__opcodes__zzabi_if)

/* `(sys__unquote 'form)` — a COPY of the form, left where the call stood to be run next: `if` with one arm and no test.
 * ⛳ `while` is NOT written with it: as a defun each turn nests inside the last, and ▶ `sys__while` holds one turn.
 * ⛔⛔ A COPY, BECAUSE RUNNING A FORM CONSUMES IT: the evaluator rewrites a form in place as it reduces it, so running the
 *   quoted form itself would leave `while`'s next turn a list already reduced — `MEASURED`, "EVAD" on the
 *   first turn. So the form is pictured and a running copy thawed from the picture, as `begin` gives a block its program;
 *   the original is untouched and may be run again. A picture is thawed as it is.
 * ⛔ ONLY A QUOTED FORM — a value is already what it is, and running it has no meaning. */
static __device__ sys__heap_node sys__opcodes__zzabi_unquote(const sys__heap_node* argv, unsigned argc,
                                                              sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t run = 0ull;
    if (argv[0].dtype == SYS__KIND__QUOTED_LIST) {
        const uint64_t picture = sys__sublist__deep_copy_to_node_array(argv[0].args[0]);
        if (picture == 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);     /* the picturing raised */
        run = sys__list__thaw_running(picture);
        (void)sys__heap_object__release(picture);          /* ⛳ after the thaw, which reads it — as `begin` does */
    } else if (argv[0].dtype == SYS__KIND__QUOTED_ARRAY) {
        run = sys__list__thaw_running(argv[0].args[0]);
    } else {
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    }
    if (run == 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__heap_node form = sys__heap_object__reference_to(run);   /* the copy's own hold, handed over */
    return form;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_unquote, sys__opcodes__zzabi_unquote)

/* ══ ⭐⭐ THE REST — the environment's pair, the dispatch trio, the dictionary, a package's init, `defun` ══
 *
 * ⛳ EACH IS AN ARITY GUARD AND A TYPE CHECK IN FRONT OF THE CALL IT NAMES. `package_init` and `defun`
 * have more to say, and say it at their own site. */

/* `(sys__bindings__viewonly env)` — a QUOTED_ARRAY of what the scopes mean, for a block to be handed.
 * ⭐⭐ IT IS PUBLISHED BECAUSE DISPATCHING NEEDS IT AND NOTHING ELSE CAN MAKE ONE. A computation sent to
 * another block is sent a SNAPSHOT: the receiving block builds its own scopes over it, so a read falls
 * through to the snapshot and a write never leaves the scopes. Handing over the environment itself would
 * have two blocks writing one scope stack.
 * ⛳ AND IT IS A PICTURE RATHER THAN A WINDOW, which is what makes it safe to hand to sixty readers:
 * nothing the source binds afterwards appears in it. */
static __device__ sys__heap_node sys__opcodes__zzabi_viewonly(const sys__heap_node* argv, unsigned argc,
                                                               sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype) || !sys__bindings__is(argv[0].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t array = sys__bindings__create_viewonly_array(argv[0].args[0]);
    /* ⛔ THE WORD SAYS THE MAKING FAILED AND NOT WHY: an environment with no symbols, a carve with no
     * room and a refused placement all answer zero, and each raised its own word where it happened — so
     * the log has the cause and the program is handed only what this site can know. */
    if (array == 0ull) return sys__engine__abi__error(SYS__BINDINGS__FAULT_NO_VIEW);
    sys__heap_node said;
    said.dtype = SYS__KIND__QUOTED_ARRAY; said.num_args = 0u; said.op_code = 0ull; said.args[0] = array;
    return said;                      /* ⛳ the maker's hold IS what the verb hands over */
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_viewonly, sys__opcodes__zzabi_viewonly)

/* `(sys__bindings__set env 'name value)` — change what a name already means. The name arrives quoted, so
 * the scanner leaves it alone and this is handed the symbol rather than what it currently means.
 * ⛳ IT ANSWERS `#t` OR THE ERROR ITSELF, AND THERE IS NO NO: every way this can fail is a refusal rather
 * than a negative result. ⛔ The error carries `SYS__BINDINGS__FAULT_NO_SCOPE` whichever refusal it met —
 * a value that is nothing, a reference that is not a bindings, a name with no scope of its own. Which of
 * them it was goes to the fault channel, raised by the set before it answers. */
static __device__ sys__heap_node sys__opcodes__zzabi_bindings_set(const sys__heap_node* argv,
                                                                   unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    /* ⛳ 0 FOR THE ENVIRONMENT IS THE ONE THIS COMPUTATION RUNS IN — ▶ `sys__bindings__zzpackage_named` */
    if (sys__bindings__set(sys__bindings__zzpackage_named(ctx->base, argv[0].args[0]), &argv[1], &argv[2]))
        return sys__opcodes__truth(true);
    return sys__engine__abi__error(SYS__BINDINGS__FAULT_NO_SCOPE);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_bindings_set, sys__opcodes__zzabi_bindings_set)

/* ── THE DISPATCH TRIO. Each addresses a block by its number. ───────────────────────────────────────── */

/* `(sys__compute block names actions)` — put a computation on another block and carry on.
 * ⭐⭐ IT IS `claim` AND `init` AND NOTHING ELSE. A claim without a fill leaves a base held by nobody; a
 * fill without a claim is the race `init` refuses. They are one word because they are never separately
 * useful, and separating them would publish the half that can strand a block.
 * ⭐ BOTH OPERANDS ARE READ-ONLY COPIES, AND THAT IS WHY THIS IS SAFE AT SIXTY BLOCKS: the receiving block
 * thaws its own program out of one and builds its own scopes over the other, so nothing either block
 * writes is anything the other can see. ⛔ An environment is refused here although the machinery beneath
 * would accept one — a program dispatching elsewhere would be handing a second block one scope stack.
 * ⛳ The actions arrive unevaluated with no marking, because a picture is a `QUOTED_ARRAY` and the scan
 * descends only into a reference naming a list.
 * ⛳ THE SHAPES ARE CHECKED BEFORE ANYTHING IS CLAIMED, so a refused compute leaves every base as it found
 * it. What is checked is what the offset NAMES, never how the cell is tagged: a picture carried inside a
 * picture comes back from the thaw as an `OBJECT_REFERENCE`, same offset, same array.
 * ⛔ IT ANSWERS A REAL YES, AND EVERY NO IS AN ERROR SAYING WHICH — a busy block, one never started, an
 * operand of the wrong shape and a fill with no room are four different repairs. */
static __device__ sys__heap_node sys__opcodes__zzabi_compute(const sys__heap_node* argv, unsigned argc,
                                                              sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[0].args[0] >= (uint64_t)sys__silicon__block_count())
        return sys__engine__abi__error(SYS__COMPUTING_BASE__FAULT_NO_BLOCK);
    sys__heap_node* there = sys__computing_base__get((uint32_t)argv[0].args[0]);
    if (there == 0 || !sys__computing_base__is(there))
        return sys__engine__abi__error(SYS__COMPUTING_BASE__FAULT_NO_BLOCK);
    const uint32_t me = (uint32_t)sys__silicon__block_id();
    if (sys__computing_base__claim(there, me) == 0)
        return sys__engine__abi__error(SYS__COMPUTING_BASE__FAULT_BUSY);
    if (!sys__computing_base__init(there, me, argv[1].args[0], argv[2].args[0])) {
        /* ⛳ A claim taken and an init refused leaves the base half-standing, so it is finished and read
         * out before the refusal goes back. */
        (void)sys__computing_base__finish(there, me, SYS__COMPUTING_BASE__ERROR, sys__heap_node__nothing());
        (void)sys__computing_base__read_result(there, me);
        return sys__engine__abi__error(SYS__COMPUTING_BASE__FAULT_HALF);
    }
    return sys__opcodes__truth(true);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_compute, sys__opcodes__zzabi_compute)

/* ⛳ `completed` AND `result` SHARE THEIR WHOLE PREAMBLE, so it is written once. It answers the base,
 * or zero with the reason in `why`. */
static __device__ inline sys__heap_node* sys__opcodes__zzabi_base_at(const sys__heap_node* argv,
                                                                      unsigned argc, uint64_t* why) {
    if (argc != 1u)                                 { *why = SYS__OPCODES__FAULT_ARITY;              return 0; }
    if (argv[0].dtype != SYS__KIND__VALUE_INT)      { *why = SYS__OPCODES__FAULT_TYPE;               return 0; }
    if (argv[0].args[0] >= (uint64_t)sys__silicon__block_count())
                                                    { *why = SYS__COMPUTING_BASE__FAULT_NO_BLOCK;    return 0; }
    sys__heap_node* there = sys__computing_base__get((uint32_t)argv[0].args[0]);
    if (there == 0 || !sys__computing_base__is(there))
                                                    { *why = SYS__COMPUTING_BASE__FAULT_NO_BLOCK;    return 0; }
    *why = 0ull;
    return there;
}

/* `(sys__completed block)` — ⚖ a real `#t`/`#f`, which is what lets a program POLL rather than block:
 * collecting is what takes the value, so without this a program could not ask without committing.
 * ⛳ IT ANSWERS WHETHER THERE IS ANYTHING LEFT TO WAIT FOR, NOT WHETHER IT WENT WELL — an errored
 * computation is over too, and what happened is the result's business. It takes nothing and changes
 * nothing, so a program may ask as often as it likes; only `sys__result` moves the base. */
static __device__ sys__heap_node sys__opcodes__zzabi_completed(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    (void)ctx;
    uint64_t why = 0ull;
    sys__heap_node* there = sys__opcodes__zzabi_base_at(argv, argc, &why);
    if (there == 0) return sys__engine__abi__error(why);
    return sys__opcodes__truth(sys__computing_base__completed(there));
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_completed, sys__opcodes__zzabi_completed)

/* `(sys__result block)` — take what that block's computation came to, and put its base back.
 * ⭐⭐ COLLECTING IS WHAT ENDS A COMPUTATION, AND NOTHING ELSE DOES. The base goes back to FREE here and
 * nowhere else, so a `compute` whose result is never taken holds that block's base forever and the next
 * `compute` onto it is refused BUSY — loudly, at the next dispatch, rather than as a silent leak.
 * ⭐ IT HANDS THE VALUE OVER RATHER THAN LENDING IT: the slot is emptied, and a second collector is
 * handed nothing.
 * ⛳ IT WAITS, BOUNDED. A result read before the block has published one is nothing, so this retries up
 * to `SYS__COMPUTING_BASE__RESULT_TRIES` with a back-off; a computation that never finishes answers
 * NOTHING rather than parking the block that asked. The non-blocking shape is `sys__completed`: ask,
 * and collect when the answer is `#t`. */
static __device__ sys__heap_node sys__opcodes__zzabi_result(const sys__heap_node* argv, unsigned argc,
                                                             sys__engine__ctx* ctx) {
    (void)ctx;
    uint64_t why = 0ull;
    sys__heap_node* there = sys__opcodes__zzabi_base_at(argv, argc, &why);
    if (there == 0) return sys__engine__abi__error(why);
    const uint32_t me = (uint32_t)sys__silicon__block_id();
    sys__heap_node got = sys__heap_node__nothing();
    for (uint64_t k = 0ull; k < SYS__COMPUTING_BASE__RESULT_TRIES; ++k) {
        got = sys__computing_base__read_result(there, me);
        if (got.dtype != SYS__KIND__VALUE_NULL || got.args[0] != 0ull) break;
        sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
    }
    return got;                       /* ⛳ read_result hands over a hold; the bridge gives it back */
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_result, sys__opcodes__zzabi_result)

/* ── THE DICTIONARY'S TWO DOORS ────────────────────────────────────────────────────────────────────── */

/* `(sys__dictionary__put dict key value)` -> `#t` / `#f`. False on a SEALED dictionary and on a heap
 * that would not carve — two different facts, each raised by the dictionary itself. */
static __device__ sys__heap_node sys__opcodes__zzabi_dict_put(const sys__heap_node* argv, unsigned argc,
                                                               sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype) || !sys__dictionary__is(argv[0].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const sys__heap_node key = sys__dictionary__zzpackage_key_of(&argv[1]);
    return sys__opcodes__truth(sys__dictionary__put(argv[0].args[0], &key, &argv[2]));
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_dict_put, sys__opcodes__zzabi_dict_put)

/* `(sys__dictionary__get dict key)` -> the value, or NULL when the key is not there.
 * ⛳ ABSENT IS A VALUE, NOT AN ERROR: asking a table about a key it does not hold is an ordinary question,
 * and a dispatch that falls through to a default needs exactly that. ⛔ A first argument that is not a
 * dictionary still refuses — that is a mistake in the program, not a fact about the table. */
static __device__ sys__heap_node sys__opcodes__zzabi_dict_get(const sys__heap_node* argv, unsigned argc,
                                                               sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype) || !sys__dictionary__is(argv[0].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const sys__heap_node key = sys__dictionary__zzpackage_key_of(&argv[1]);
    return sys__dictionary__get(argv[0].args[0], &key);   /* held; the bridge gives the hold back */
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_dict_get, sys__opcodes__zzabi_dict_get)

/* `(sys__node_array__get array i)` -> element `i`, by value. ⛳ A GET TAKES A HOLD, per the array's own
 * rule, and the bridge gives the verb's back once the form has taken its own. ⛔ An index past the end
 * refuses: an array's length is fixed, so asking past it is a mistake in the program. */
static __device__ sys__heap_node sys__opcodes__zzabi_array_get(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype) || !sys__node_array__is(argv[0].args[0])
     || argv[1].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[1].args[0] >= sys__node_array__length(argv[0].args[0]))
        return sys__engine__abi__error(SYS__NODE_ARRAY__FAULT_RANGE);
    return sys__node_array__get(argv[0].args[0], argv[1].args[0]);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_array_get, sys__opcodes__zzabi_array_get)

/* `(sys__node_array__set array i value)` -> `value`. ⛳ THE ARRAY TAKES ITS OWN HOLD on what it stores and
 * gives up the one on what was there (`sys__node_array__set`), so a program can fill a table and let go of
 * everything it made along the way. ⛔ An index past the end refuses, as the read does. */
static __device__ sys__heap_node sys__opcodes__zzabi_array_set(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype) || !sys__node_array__is(argv[0].args[0])
     || argv[1].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    if (argv[1].args[0] >= sys__node_array__length(argv[0].args[0]))
        return sys__engine__abi__error(SYS__NODE_ARRAY__FAULT_RANGE);
    if (!sys__node_array__set(argv[0].args[0], argv[1].args[0], &argv[2]))
        return sys__engine__abi__error(SYS__NODE_ARRAY__FAULT_RANGE);
    return sys__node_array__get(argv[0].args[0], argv[1].args[0]);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_array_set, sys__opcodes__zzabi_array_set)

/* `(sys__file__open 'path)` -> a handle, an integer never zero. The path is a quoted name — spelled out as
 * its string when it is too long to travel as a number, and looked up when it travels as one. The worker's
 * family opens it: its `file_read` is what later puts the bytes in its memory. */
#define SYS__FILE__PATH_MAX 4096u
static __device__ sys__heap_node sys__opcodes__zzabi_file_open(const sys__heap_node* argv, unsigned argc,
                                                                sys__engine__ctx* ctx) {
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    sys__heap_node spelled = sys__heap_node__nothing();
    uint64_t string = 0ull;
    if (argv[0].dtype == SYS__KIND__QUOTED_STRING_NAME) {
        string = argv[0].args[0];
    } else if (argv[0].dtype == SYS__KIND__QUOTED_NAME) {
        spelled = sys__symbol__name(argv[0].args[0]);           /* held; given back below */
        string = spelled.dtype == SYS__KIND__OBJECT_REFERENCE ? spelled.args[0] : 0ull;
    }
    char path[SYS__FILE__PATH_MAX];
    const bool named = string != 0ull && sys__string__is(string) && sys__string__length(string) < SYS__FILE__PATH_MAX;
    if (named) {
        const uint64_t n = sys__string__length(string);
        const uint8_t* bytes = sys__string__bytes(string);
        for (uint64_t i = 0ull; i < n; ++i) path[i] = (char)bytes[i];
        path[n] = '\0';
    }
    if (sys__heap_node__carries_reference(spelled.dtype)) (void)sys__heap_object__release(spelled.args[0]);
    if (!named) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t handle = 0ull;
    if (!sys__gpu__file_open(ctx->family, path, &handle)) return sys__engine__abi__error(SYS__FILE__FAULT_OPEN);
    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT; answer.args[0] = handle;
    return answer;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_file_open, sys__opcodes__zzabi_file_open)

/* `(sys__file__close handle)` -> nothing. */
static __device__ sys__heap_node sys__opcodes__zzabi_file_close(const sys__heap_node* argv, unsigned argc,
                                                                 sys__engine__ctx* ctx) {
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[0].args[0] == 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__gpu__file_close(ctx->family, argv[0].args[0]);
    return sys__heap_node__nothing();
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_file_close, sys__opcodes__zzabi_file_close)

/* `(sys__worker)` -> the worker this runner is: its block's, or worker 0 past the last. */
static __device__ sys__heap_node sys__opcodes__zzabi_worker(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    (void)argv; (void)ctx;
    if (argc != 0u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    sys__heap_node n = sys__heap_node__nothing();
    n.dtype = SYS__KIND__VALUE_INT; n.args[0] = (uint64_t)sys__silicon__worker();
    return n;
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_worker, sys__opcodes__zzabi_worker)

/* `(sys__package__init '(the list of inits))` — the language's own init.
 * ⛳ THE ARGUMENT IS CHECKED THOUGH NOTHING IS DONE WITH IT, because the refusal is what says this call
 * was built the way every other init's is — in the one package guaranteed to be in every list.
 * ⭐ ONE THING TO DO, AND IT IS THE ONE THING THE MACHINE NEEDS A HEAP FOR. This call is being evaluated,
 * so the heap, the register, the blocks and the evaluator are all standing. The symbol tables are
 * dictionaries, so they can only be made once there is a heap — and every other package's init may want
 * a name looked up, which is why theirs wait for this one.
 * ⛔ THE FAULT CODES ARE THE PACKAGE'S, not the opcodes' — a package's init refuses in its own
 * vocabulary. */
static __device__ sys__heap_node sys__opcodes__zzabi_package_init(const sys__heap_node* argv,
                                                                   unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__PACKAGE__FAULT_ARITY);
    /* ⛳ IT ARRIVES QUOTED, because an object reference naming a list is something the scan REDUCES. */
    if (argv[0].dtype != SYS__KIND__QUOTED_LIST || argv[0].args[0] == 0ull)
        return sys__engine__abi__error(SYS__PACKAGE__FAULT_LIST);
    if (!sys__symbol__zzpackage_stand_up()) return sys__engine__abi__error(SYS__SYMBOL__FAULT_NO_TABLE);
    return sys__heap_node__nothing();
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_package_init, sys__opcodes__zzabi_package_init)

/* ── A PROCEDURE, AND WHAT MAKES ONE ─────────────────────────────────────────────────────────────────
 * A procedure is a PICTURE of two things: the names its arguments are bound to, and the body to run —
 * plain arrays all the way down, which nothing writes once `defun` has made them. It is an ordinary object
 * of an ordinary kind, so it is held, released and put in a binding like any value — ⚖ and a `defun`'s
 * destination is LEXICAL, so what `defun` does is BIND A NAME. Nothing here has a table of functions; a
 * function is what a name means.
 * ⭐ A PICTURE BECAUSE EVERY CALL READS A PROCEDURE AND NONE WRITES IT. A call makes a list of its own out
 * of the body and runs that, and the picture is read without a hold — so a binding every block shares
 * hands each of them a procedure it can copy with nothing to wait on.
 * ⛳ A PROCEDURE OF NO PARAMETERS HAS NIL WHERE THEIR NAMES WOULD BE, because an empty list is nil in a
 * picture.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ⛳ THE LAYOUT IS DESCRIBED IN `procedure.cuh`, WITH THE KIND — this file writes the two halves and
 * reads what they mean from where the thing itself is described. */

/* `(defun env 'name '(params) '(body))` — make a procedure and bind it.
 * ⛳ THE PARAMETERS AND BODY ARRIVE QUOTED, for the reason every unevaluated thing here does — the scan
 * does not descend into a value, so a body is not run at the moment it is written down. */
static __device__ sys__heap_node sys__opcodes__zzabi_defun(const sys__heap_node* argv, unsigned argc,
                                                            sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    /* ⛳ 0 FOR THE ENVIRONMENT IS THE ONE THIS COMPUTATION RUNS IN — ▶ `sys__bindings__zzpackage_named` */
    const uint64_t       env    = sys__bindings__zzpackage_named(ctx->base, argv[0].args[0]);
    const sys__heap_node name   = argv[1];
    const sys__heap_node params = argv[2];
    const sys__heap_node body   = argv[3];
    /* ⭐⭐ NIL IS ACCEPTED WHERE THE PARAMETERS GO, AND THAT IS WHAT A PROCEDURE OF NO PARAMETERS IS.
     * ⚖ nil IS the empty list, so the door agrees with what is already true inside: a picture turns an
     * empty list into nil, and refusing it here made the going-in disagree with the coming-out.
     * ⛳ THE BODY IS NOT GIVEN THE SAME LATITUDE — there is no reading of nil that is a thing to do. */
    bool params_ok = params.dtype == SYS__KIND__QUOTED_LIST
                  || params.dtype == SYS__KIND__VALUE_NULL;
    const uint64_t arity = params.dtype == SYS__KIND__QUOTED_LIST
                         ? sys__sublist__length(params.args[0]) : 0ull;
    for (uint64_t i = 0ull; params_ok && i < arity; ++i)
        params_ok = sys__bindings__quoted(sys__sublist__nth(params.args[0], i).dtype);
    if (!params_ok || !sys__bindings__quoted(name.dtype) || body.dtype != SYS__KIND__QUOTED_LIST)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);

    /* The procedure is the picture of `(params body)`: the pair is only the way there and goes at once. */
    const uint64_t pair = sys__list__create();
    if (pair == 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    (void)sys__sublist__append(pair, &params);
    (void)sys__sublist__append(pair, &body);
    const uint64_t proc = sys__sublist__deep_copy_to_node_array(pair);
    (void)sys__heap_object__release(pair);
    if (proc == 0ull) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);

    /* ⭐ TAGGED, NOT SHAPED. */
    sys__procedure__zzpackage_adopt(proc);
    sys__heap_node value = sys__heap_object__reference_to(proc);
    value.dtype = SYS__KIND__PROCEDURE_REFERENCE;
    const bool ok = sys__bindings__add(env, &name, &value);
    (void)sys__heap_object__release(proc);            /* the binding holds it now */
    return sys__opcodes__truth(ok);
}
SYS__ENGINE__ABI__BRIDGE(sys__opcodes__zzabi_adapter_defun, sys__opcodes__zzabi_defun)

#endif /* SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_ABI_CUH */
