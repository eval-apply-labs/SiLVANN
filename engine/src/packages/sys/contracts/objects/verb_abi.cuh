#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_VERB_ABI_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_VERB_ABI_CUH

/* This file needs nothing: every name in it is its own. */

/* ══ ⛔⛔⛔ THE CEILING — AND IT IS A BOUND ON *FIXED-ARITY* VERBS ONLY, WHICH IS THE WHOLE PROBLEM ═══
 *
 * The census puts the widest FIXED-arity verb at 9 (`nn__attention__decode`), and the ceiling is 12 so the
 * next wider verb does not meet it the day it is written; on the host it is an array of 12 nodes, 768 bytes
 * of stack. ⛔ BUT THE LANGUAGE HAS
 * VARIADIC VERBS AND THEY HAVE NO BOUND AT ALL: `begin`, `prog1`, `remove_bindings` and `let` take as
 * many cells as a program hands them, and the composer really does emit wide ones.
 *
 * ⛔⛔ `MEASURED`, AND IT IS A DESIGN LIMIT RATHER THAN A TUNING KNOB:
 * ```
 *   a real program:  (sys__begin <form> x 400)          test/src_host_eval_rate.cpp
 *   MAX_ARGV = 8     the bridge REFUSES it as an arity fault; the program does not run
 *   MAX_ARGV = 512   it runs — at 32 KB OF STACK PER VERB CALL (a node is 64 B)
 * ```
 * ⇒ ★★ **A FIXED CEILING CANNOT HOLD A VARIADIC VERB.** Materialising N operands costs O(N) stack and
 *   O(N) locates, and there is no N that is both large enough for a program and cheap for a verb that
 *   reads one cell. ⇒ THE FOUR VARIADIC VERBS DO NOT GO THROUGH THIS BRIDGE; they keep the form-reading
 *   shape and read only the cells they use.
 *
 * ⛔ WHAT IT COSTS TO BRIDGE ONE ANYWAY, MEASURED SO THE CHOICE IS PRICED: with the cap raised, the host
 * evaluator's per-form rate went **0.3454 -> 0.3751 us, +8.6%**, five reps each and NO OVERLAP between
 * the two sets. The bridge turns `begin`'s single `nth(n-1)` into N `nth` calls — O(N) locates for a
 * verb that reads one cell.
 * ⚠ AND NEITHER BEHAVIOUR SUITE BUILDS A `begin` WIDE ENOUGH TO HIT THE CAP — only the rate probe does. */
/* ⭐ A DOOR ON THE WORKER'S CARD FLAGGED ITS FAULT WORD WITHOUT NAMING THE FAULT. Doors write into a word
 * on the worker's own device — the fault's own four-byte word where they have one, as an overflow does;
 * when a program's answer is collected the engine reads it and raises it ONCE into the machine's channel,
 * which is how a worker's fault cascades to whoever asked. A bare flag of 1 is raised as this. */
#define SYS__VERB_ABI__FAULT_WORKER  0x53455746ull   /* "SEWF" */

#ifndef SYS__ENGINE__ABI__MAX_ARGV
#define SYS__ENGINE__ABI__MAX_ARGV 12u
#endif

#include <stdint.h>
#include "heap_node.cuh"
#include "../abi/silicon_family.cuh"   /* what a family is, which a verb launches on */

/* What a verb is given besides its operands. ⛳ `sys` ignores it; `nn` launches with it. */
typedef struct {
    sys__silicon_family__id family;   /* the silicon whose family this verb's doors are on — the worker's own;
                                  SYS__SILICON_FAMILY__NONE where no worker has bound one */
    unsigned  device;
    void*     stream;
    uint32_t* fault_word;      /* ⚖ "start with the word" — atomicOr'd, read at a sync point */
    /* ⛔⛔ THE ARENA — the computing base a verb carves a new object out of. `MEASURED` over every
     * `sys` verb body in this shape (`grep -n 'ctx->' cpu/opcodes/opcodes_abi.cuh`): all but two ignore
     * it, and those two, `sys__clone` and `sys__create`, USE it.
     * ⇒ ★★ THE EXCEPTION IS NOT A LONG TAIL, IT IS THE ALLOCATORS. "All but two ignore it" reads as a
     *   rounding error; the two are every verb in the tier that makes something.
     * ⛳ AND `nn` WANTS IT FOR THE SAME REASON THE MOMENT A BUFFER COMES FROM THE POOL rather than
     *   from an operand, so this is not a `sys` accommodation. */
    sys__heap_node* base;
    /* ⭐ THE HEAP AS THE WORKER'S CARD SEES IT — where a kernel may write a value straight into a node the
     *   program made, ⚖ *"allocate the return nodes beforehand and send the node array base address"*. The
     *   heap is registered with the card at boot, and `heap_card` is its first node in the card's view; a
     *   node at `p` is at `heap_card + (p - heap_host)` there — counted in nodes, like every address in a
     *   package. Zero where the family could not register it (OpenCL on the MI50), and a verb copies. */
    sys__heap_node* heap_host;
    sys__heap_node* heap_card;
    uint64_t        heap_nodes;
} sys__engine__ctx;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_VERB_ABI_CUH */
