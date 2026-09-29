#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PRIMITIVES_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PRIMITIVES_CUH

/* This file needs nothing: every name in it is its own. */
#define NN__PRIMITIVES__FAULT_BOUNDS 0x4E50424Eull   /* "NPBN" — a count past what a buffer holds */
#define NN__PRIMITIVES__FAULT_NO_DEVICE 0x4E504E44ull   /* "NPND" — No Device: no runtime's doors on
                                                          the verb's silicon, or no fault word to raise into */

/* ══ ⭐⭐⭐ "NPOV" — A VALUE TOO LARGE FOR THE HALF IT IS BEING STORED IN ═════════════════════════════
 * ⚖ ARCHITECT: *"i think we hace been bitten in the past due to overflows."* ⇒ *"yes arm
 * it."*
 *
 * ⛔⛔ THE HEADROOM IS LARGE AND THAT IS NOT WHY THIS EXISTS. `MEASURED` on the real 35B over 256
 * tokens of unseen text, every module output: the worst peak activation is **109.0** against fp16's
 * ceiling of **65504** — 601x of headroom, and 0 of 634 modules anywhere near it.
 * ⇒ ★★ AND THE PROBE IS WEAKER THAN THE CLAIM IT WOULD SUPPORT, IN THREE WAYS: it is ONE input; it
 * is 256 tokens against a design context of 256k, and residual-stream peaks grow with length; and it
 * hooks module OUTPUTS, not the intermediates inside them — `swiglu`'s own ±88 clamp exists because
 * `expf` overflows, which is exactly such an intermediate.
 * ⇒ ⭐ SO THE MARGIN IS NOT THE ANSWER. `float_to_half` already BRANCHES on this case and silently
 * answers `inf`, which then propagates through every remaining layer and arrives as a wrong answer
 * with no location. The branch is already taken; making it SAY SO costs nothing on the hot path.
 * ⛳ THE SAME ARGUMENT `swiglu__impl` MAKES ABOUT ITS OWN BOUND: *"the failure mode is not a fault,
 * it is a healthy-looking answer — which is why the check is here and not left to the hardware:
 * there is no hardware event to leave it to."* An fp16 overflow is that shape exactly.
 *
 * ── ⚖⚖ *"inf for the value, keep the raise."* ───────────────────────────────────────────────────────
 * ⛔⛔ "THROWING" HAS TWO HALVES, AND ONCE A LOOP IS A KERNEL ONLY ONE OF THEM IS CHEAP. Run as a
 * launch, a body writes `over` to device memory, and reading it back costs a SYNCHRONISATION —
 * `MEASURED` gfx906: **1.654 µs to launch and keep going against 28.816 µs to launch, sync, read the
 * word back and branch. 17.4x**, and it would be paid by every buffer verb on every call whether or not
 * anything ever overflows.
 * ```
 *   the VALUE becoming an error   ⇒ NOT DONE. It is what `is_error` would test, and it is what costs
 *                                   a per-verb sync. The buffer keeps `inf`, which is what
 *                                   `float_to_half` already puts there and what IEEE says it is.
 *   the RAISE into the channel    ⇒ DONE. One write on a branch already taken, and the channel is
 *                                   read host-side at the apply boundary — a sync already paid for.
 * ```
 * ⇒ ⭐ **THE LOCALISATION THE PARAGRAPH ABOVE ASKS FOR IS THE RAISE, NOT THE VALUE.** *"arrives as a
 * wrong answer with no location"* is answered by the log; no program branches on an overflow, so
 * nothing needs the error object. ⛳ `error.cuh` already draws this line — *"the raise puts it in the
 * log where somebody watching a run will see it, the object puts it in the hands of whoever asked"* —
 * and for this fault only the first reader exists.
 * ⛳ **A REFUSAL WOULD NOT ADD A RAISE — IT WOULD ADD A HEAP ALLOCATION:** `sys__opcodes__fails`
 * reaches `sys__error__make`, which raises the same fault, so the 18 sites that raise directly skip
 * only the refusal and the object it allocates.
 *
 * ⚠ **TWO CONSEQUENCES, NEITHER HIDDEN:**
 * ① `fault__impl.cuh`'s `__noinline__` rests on *"every raise is a refusal and every refusal returns"*,
 *   and these 18 raises fall through. Its COLD-PATH argument holds — this is one raise per VERB CALL,
 *   not per element, on a path `MEASURED` at 601x of headroom — but the stated falsifier is partly met
 *   and says so at its own site.
 * ② The channel is SINGLE-SLOT and LAST-WINS. An overflow does not stop the program, so a later fault
 *   overwrites its word. `raised` COUNTS, so a run reports how many happened and the last one's
 *   identity — the same shape any multi-fault run has. */
#define NN__PRIMITIVES__FAULT_OVERFLOW 0x4E504F56ull  /* "NPOV" */

/* ══ ⭐⭐⭐ AN ELEMENT IS TWO BYTES — ⚖ RULED ═══════════════════════════════════════════════
 *
 * ⚖ ARCHITECT: *"lets move to a fp16 setup"* … *"b, but keep the fp32 accumulator"*.
 *
 * ⭐⭐ THE WHOLE LIBRARY'S STORAGE TYPE IS THIS ONE CONSTANT AND THE CASTS THAT READ IT, AND THAT IS
 *   WHY THE CHANGE WAS CHEAP: `nn__buffer` has NEVER known its element type — `room` answers an
 *   address and a BYTE count, and each verb casts for itself. `MEASURED` at the change: `float`
 *   appears in exactly four impl files and ZERO times in `buffer__{header,impl}.cuh`.
 *   ⇒ ★★ THE POOL DID NOT MOVE. The same property that let one `else if` in `room` give every compute
 *   verb pages let one constant here give every compute verb halves.
 *
 * ⛔⛔ STORAGE IS fp16; ARITHMETIC IS NOT. Every verb loads to `float`, computes in `float` and
 *   converts once on the way out. ⚖ *"keep the fp32 accumulator"* — a 2048-term reduction in fp16
 *   loses more to its accumulator than q8 loses to quantisation, which would make the whole width
 *   ladder unreadable. The rule is: **the array is half, the register is float.**
 *
 * ⛳ AND THE ONE PLACE THAT RULE IS NOT OBVIOUSLY SATISFIABLE IS THE BUTTERFLY, WHERE THE ACCUMULATOR
 *   *IS* THE ARRAY — `nn__hadamard__zzabi_body_rotate` rewrites its output `log2(n)` times, so
 *   fp16 storage rounds eleven times at n=2048. `MEASURED` rather than argued: **0.07%
 *   relative error on four real q_proj rows, against 9.7% for q4 itself** — 1/140th of the error
 *   budget it sits inside. ⇒ ★ IT NEEDED NO fp32 SCRATCH, and the reason to record the number is that
 *   the obvious remedy (a scratch array of `n` floats) would have cost a span allocation for nothing.
 */
#define NN__PRIMITIVES__ELEMENT_BYTES 2ull

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_PRIMITIVES_CUH */
