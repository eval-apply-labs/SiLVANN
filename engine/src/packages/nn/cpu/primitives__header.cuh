#ifndef SILVANN__PACKAGES_NN_CPU_PRIMITIVES__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_PRIMITIVES__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../../sys/contracts/objects/kind.cuh"              /* what a thing IS — the first word of every node */
#include "../../sys/cpu/heap_node__header.cuh" /* the node a form's arguments arrive in */
#include "../contracts/objects/primitives.cuh" /* its constants and fault words */
/* ══ nn — THE ARITHMETIC PRIMITIVES, AND THE ONE RESOLUTION THEY ALL SHARE ═══════════════════════════
 *
 * ⚖ ARCHITECT: *"lets go ahead with the simple opcodes."* Three of them, and they are not
 * three of a kind — each is a different SHAPE, which is why these three and not the three smallest:
 * ```
 *   pointwise add   n in, n out, no communication       the residual connection
 *   rmsnorm         n in, ONE number, then n out        a REDUCTION and a broadcast back
 *   argmax          n in, ONE number out                a reduction that answers a NUMBER, not a buffer
 * ```
 * ⇒ ⭐⭐ **ARGMAX IS THE STRATEGIC ONE AND IT IS NOT THE ARITHMETIC.** It is the first verb in this tree
 * that answers an INTEGER rather than an object, which is exactly what a decode loop needs: `poll` hands
 * a program's integer answer straight back to the host, so a token id needs no buffer, no address and no
 * DMA. ⇒ ★ THE LAST LINK OF A GENERATE LOOP IS A VERB THAT RETURNS A NUMBER.
 * ⛳ AND ITS GRID RENDEZVOUS DISSOLVED. The old `NATIVE_ARGMAX` is a two-barrier op — every block writes
 * a partial, block zero folds `gridDim.x` of them — and it was one of only TWO grid syncs to survive the
 * census. Run as one block of one thread (`gpu/doors.cuh`) there is nobody to rendezvous with, so the op is a loop.
 * ⚠ THAT IS A PROPERTY OF HOW IT IS CALLED, NOT A REFUTATION OF THE BARRIER: the day this runs grid-wide
 * the fold comes back. It is correct now and it is not parallel, and those are different claims.
 *
 * ── ⭐ WHY THE RESOLVER LIVES HERE AND NOT IN EACH OP ────────────────────────────────────────────────
 * Every one of these takes buffers and a count, and every one must answer the same two questions about
 * each: WHERE are its bytes, and HOW MANY are there. Written per op that is four copies of a lookup and
 * four chances to check a bound one way in one place and another way elsewhere. ⇒ ★ THE SHARED THING IS
 * NOT THE ARITHMETIC, IT IS THE ARGUMENT HANDLING — which is the half that repeats and the half a reader
 * stops reading after the second time.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* A buffer argument, from a reference to a place and the room it names. ⛳ THE CLASS IS READ, NOT
 * ASSUMED: a buffer knows which class it came from and the class knows its bytes, so the bound is the
 * pool's own figure rather than a number these files would have to be told. */
static __device__ inline bool nn__primitives__room(const sys__heap_node* ref,
                                                    uint64_t* at, uint64_t* bytes);
/* ⛳ THE SAME, WHATEVER WORKER OWNS IT — for the one verb whose job is to cross: `nn__buffer__copy`'s source. */
static __device__ inline bool nn__primitives__room_of_any(const sys__heap_node* ref,
                                                          uint64_t* at, uint64_t* bytes);
/* The owning worker plus one, of a buffer or a weights view, or 0. */
static __device__ inline uint64_t nn__primitives__owner(const sys__heap_node* ref);

/* ⛔ `count * NN__PRIMITIVES__ELEMENT_BYTES` FITS THE ROOM — and the multiply is checked for wrap first,
 * because a count near 2^63 would otherwise come back INSIDE the room after overflowing. ⇒ ★ A BOUND
 * CHECKED ON A WRAPPED PRODUCT IS NOT A BOUND, and it is the shape that reads as careful while admitting
 * anything.
 * ⛔ THE WIDTH IS NAMED AND NOT SPELLED OUT, because this sentence said `count * 4` while the constant
 * read `2` — a figure left behind by the fp16 ruling that moved the element under it. ⇒ ★ A COMMENT
 * THAT RESTATES A CONSTANT IS A SECOND DEFINITION OF IT, AND THE TWO DRIFT IN THE DIRECTION NOBODY
 * RE-READS. ⛳ IT ALSO MATTERS FOR A READER SIZING SOMETHING WIDER: `fits` IS fp16's bound and nothing
 * else's — the DeltaNet state is fp32 and carries its own, for exactly this reason. */
static __device__ inline bool nn__primitives__fits(uint64_t n, uint64_t room);

/* ⭐ A VERB'S PLANE TABLE — the node array a model's site reads its weights and widths from: its first `planes` cells
 * buffers (each an address and its room into `at` and `room`), then integers up to `reals` (into `v`), then floats up to
 * `length` (into `r`, which may be 0 when there are none), each at its own index. False on any cell that is not what
 * its place says, or a table shorter than `length`. nn's own sites read theirs through it, and a model package's. */
static bool nn__primitives__table(uint64_t table, unsigned planes, unsigned reals, unsigned length,
                                  uint64_t* at, uint64_t* room, uint64_t* v, float* r);
/* An optional integer cell past a table's fixed length: true when the table has it and it is 1. */
static bool nn__primitives__table_flag(uint64_t table, unsigned index);

/* ══ ⭐⭐ THE STORAGE TYPE, AND THE TWO CONVERSIONS EVERY VERB IN THE PACKAGE GOES THROUGH ════════════
 * ⚖ RULED: *"lets move to a fp16 setup"* … *"b, but keep the fp32 accumulator"*.
 * **The array is half; the register is float.** Every compute verb loads through `half_to_float`,
 * computes in `float`, and converts once on the way out. ⛳ They live HERE rather than in the file
 * that first needed them (`vbr__impl`) because every verb needs them now and `primitives__impl` is
 * included ahead of `swiglu`, `vbr` and `hadamard` in the package manifest.
 * ⛔ THE TWO DO NOT SHARE AN ARGUMENT: the read is EXACT and the write ROUNDS. ▶ `primitives__impl`,
 * which carries why that still keeps both out of the seam — and why the old reason does not. */
static __device__ inline float    nn__primitives__zzpackage_half_to_float(uint16_t h);
/* ⛔ `over` IS A LOCAL FLAG, NOT A RAISE, AND THE DIFFERENCE IS THE WHOLE DESIGN. A conversion deep
 * in a loop has no `form` to fail, and raising per ELEMENT would put thousands of events in a channel
 * whose own lesson is that **a raise outlives the call** — a later check about something else then
 * reads a non-zero fault count. So the converter SIGNALS and the VERB raises, exactly once, after
 * its loop. ⇒ ★ WHERE A FAULT IS RAISED IS PART OF ITS MEANING: one per call is an answer, one per
 * element is weather. */
static __device__ inline uint16_t nn__primitives__zzpackage_float_to_half(float f, bool* over);

/* `(nn__vector__add a b out n)` — `out[i] = a[i] + b[i]`, answering OUT so a program can chain it.
 * ⛳ THE OLD OP ACCUMULATED IN PLACE (`dst[i] += src[i]`) because it WAS the residual connection and had
 * no way to name a destination. Naming all three makes the accumulate a CHOICE — pass the same buffer as
 * `a` and `out` — instead of the only thing it can do. */

/* `(nn__vector__pointwise_mul a b out n)` — `out[i] = a[i] * b[i]`, the elementwise product.
 * ⛳ IT HAS NO CALLER IN THE BUILT MODEL. The MoE block's `sigmoid(shared_expert_gate(h)) *
 * shared_expert_output` is a SCALAR times a vector — `shared_expert_gate` is `nn.Linear(hidden, 1)`,
 * so that sigmoid is one number and the `*` is a broadcast. `nn__vector__scale` is what serves it.
 * ⇒ ★ A PYTORCH `*` IS NOT EVIDENCE OF AN ELEMENTWISE OPERATION — check the shape either side of it.
 * ⚖ THE VERB IS KEPT AND FLAGGED: elementwise multiply is an ordinary primitive, and whether an
 * uncalled one stays is the architect's. ▶ `docs_history_internal/nn_retired_symbols.md`.
 * ⛳ `swiglu` ALREADY CONTAINS THIS PRODUCT AND CANNOT LEND IT: it computes `silu(gate) * up`, so a
 * caller wanting only the product would have to pass a `gate` whose silu is the identity, and there
 * is none. ⇒ ★ A FUSED OPERATION IS NOT AVAILABLE IN PARTS TO A CALLER THAT WANTS ONE PART.
 * ⛔ AND IT IS NOT THE VERB THE ROUTED SUM REALLY WANTS. Combining eight experts is
 * `out += w_j * y_j` — a SCALED ACCUMULATE with a scalar — and doing it with this verb costs a whole
 * buffer filled with one repeated number, per expert, per token. ▶ `NN-8`, which prices the
 * alternative; this one ships because the gate needs it either way and its definition is not a
 * design question. */

/* `(nn__vector__dot_product a b n)` — `Σ a[i]·b[i]`, answered as a REAL NUMBER.
 * ⚖ NAMED alongside the rename. ⛔⛔ AND IT IS A DIFFERENT OPERATION FROM THE ONE ABOVE,
 * not another name for it: a pointwise product answers a VECTOR of `n` values, a dot product answers
 * ONE. They share a multiply and nothing else. ⇒ ★ TWO OPERATIONS THAT DIFFER IN THE SHAPE OF THEIR
 * ANSWER ARE TWO VERBS, whatever they are called in conversation.
 * ⛳ IT ANSWERS A `sys__value_float` AND COULD NOT HAVE, A DAY AGO. With no float in the language the
 * only place for one number was a buffer — and a one-element buffer as a return value is a verb
 * apologising for its type system. ⛳ THE ACCUMULATOR IS `float` and the answer widens to fp64 once at
 * the end: the sum is taken at this package's usual width, and the cell simply holds it exactly. */

/* `(nn__matrix__matvec_transposed W x out rows cols)` — `out[c] = Σ_r W[r][c]·x[r]`, fp16 in and out.
 * ⚖ ASKED FOR: *"i want it to have the option to multiply a vector for a matrix and then
 * multiply it for its transposed matrix."* ⭐⭐ AND IT NEEDS NO TRANSPOSE AT ALL: `W` is read in the
 * layout it is already stored in, `[rows][cols]`, and only the ACCUMULATION changes. Materialising
 * `Wᵀ` first would cost `rows*cols*2` bytes and a whole extra pass for the same answer.
 * ⇒ ★ A TRANSPOSE IS A CHANGE OF TRAVERSAL BEFORE IT IS A CHANGE OF STORAGE.
 *
 * ⛔⛔⛔ IT IS fp16 ONLY, AND THAT IS A CORRECTNESS BOUND RATHER THAN A MISSING FEATURE.
 * ⚖ THE TWO STORAGE SHAPES, RULED: *"we only reason in float fp16 or turboquant terms at
 * steady compression per table."* `MEASURED` on the shipping bundle and it says the same — no per-row
 * rate plane anywhere, one `d` per record, and exactly two widths across 1,045 records: `{8: 528,
 * 16: 517}`.
 * ⇒ AT fp16 A HALFWORD IS A WEIGHT and this verb is meaningful. AT A TURBOQUANT WIDTH IT IS NOT, for a
 * reason the width has nothing to do with: every row is ROTATED along its CONTRACTION AXIS and carries
 * its own scale. Transposing makes the contraction run ACROSS rows, so each term would arrive from a
 * different row's rotation and a different row's scale.
 * `MEASURED` on `shared_expert.gate_proj` — a transposed matvec over the stored bytes is
 * **132% wrong**, not slightly wrong:
 * ```
 *     W^T y   |max| 1.1157        the true product
 *     R^T y   |max| 1.0003        the same walk over what the card actually holds
 *     relative error 1.325
 * ```
 * ⇒ ★★ THE CODEC'S ROTATION IS BOUND TO THE CONTRACTION AXIS, SO TRANSPOSING IS NOT A FREE CHANGE OF
 * VIEW — it is a different matrix. A transposed product over quantised weights needs them un-rotated
 * first, which is a decode, and nothing here does it silently. */

/* `(nn__vector__scale_at x alpha which out n)` — `out[i] = x[i] * alpha[which]`.
 * ⚖ NAMED BY THE ARCHITECT: *"nn__vector__scale is the right verb."* ▶ `NN-8`, which this
 * closes: combining the routed experts is `out += w_j * y_j` with `w_j` a SCALAR, and doing that with
 * `pointwise__mul` cost a buffer filled with one repeated number per expert per token.
 *
 * ⛔⛔ THE FACTOR ARRIVES IN A BUFFER AND NOT AS A LITERAL, AND THAT IS FORCED RATHER THAN CHOSEN:
 * `MEASURED` — this language's value kinds are `int`, `true`, `false`, `null` and `error`. **THERE IS
 * NO FLOAT LITERAL**, so a scalar cannot be written into a form at all. ⇒ ★ WHEN A LANGUAGE CANNOT
 * SPELL A VALUE, THE VERB MUST TAKE A PLACE TO READ IT FROM.
 *
 * ⛳ AND `which` IS WHY IT IS AN INDEX RATHER THAN JUST `alpha[0]`: the caller that wants this already
 * HAS its scalars in a buffer — the router's top-k weights are `k` numbers side by side — so an index
 * lets expert `j` be scaled by `scores[j]` with nothing copied anywhere. With `alpha[0]` alone, every
 * iteration would need a scratch buffer and a write to it, which is the cost this verb exists to
 * remove, one size smaller. ⛳ `which = 0` is the ordinary case and reads as "the scalar in there".
 * ⛔ `alpha` MAY BE `out` OR `x`: the factor is read ONCE, before the loop. */

/* `(nn__vector__scale x alpha out n)` — `out[i] = x[i] * alpha`, with `alpha` a REAL NUMBER.
 * ⚖ ASKED FOR: *"get the scalar version too."* ⛳ AND IT IS ONLY EXPRESSIBLE NOW: this verb
 * could not exist a day ago, because the language had no way to spell a number. The float kind is
 * what unlocks it — ▶ `sys__value_float`.
 * ⇒ ⭐ THE TWO ARE NOT REDUNDANT, THEY ANSWER DIFFERENT QUESTIONS. This one is for a factor a PROGRAM
 * knows — a temperature, a 1/sqrt(d), a gate. `_at` is for one it must READ, because the factor was
 * computed into a buffer by something upstream — the router's top-k weights are eight numbers side by
 * side, and nothing has to copy them out to use them.
 * ⛳ `x` MAY ALIAS `out`: the factor is a value in a register before the loop starts. */

/* `(nn__matrix__transpose src dst rows cols)` — `dst[c][r] = src[r][c]`, fp16 throughout.
 * ⚖ ASKED FOR: *"a transpose method in nn where you pass a table and it gives you its
 * transposed one."* ⛳ "table" here is a MATRIX, which is the word this package already uses — ▶ the
 * type table's *"one collection per matrix type"*.
 * ⛳ `src` MAY BE A `nn__weights` PLANE, so a resident matrix transposes straight out of the arena
 * into a buffer — `nn__primitives__room` answers for both kinds.
 * ⛔⛔ BUT ONLY WHERE THE PLANE'S BYTES ARE VALUES. At fp16 they are weights and this is meaningful;
 * at a TurboQuant width they are PACKED CODES, several to a halfword, and rearranging them produces a
 * buffer that decodes to nothing. ⇒ ★ A VERB THAT MOVES HALFWORDS IS ONLY A MATRIX
 * OPERATION WHERE A HALFWORD IS A MATRIX ELEMENT. Decode first, or use
 * `nn__matrix__matvec_transposed`, which needs no transpose at all.
 *
 * ⛔⛔ IT REFUSES TO WORK IN PLACE, AND THAT IS A CONTRACT RATHER THAN A LIMITATION. A square matrix
 * transposes in place by swapping pairs; a rectangular one CANNOT — it needs a permutation-cycle walk,
 * which is a different algorithm with different arithmetic. Offering in-place only when `rows == cols`
 * would make the verb's behaviour depend on numbers a reader cannot see at the call site.
 * ⇒ ★ A VERB WHOSE ALGORITHM CHANGES WITH ITS ARGUMENTS IS TWO VERBS WEARING ONE NAME. The caller
 * passes a second buffer, which it has. */

/* `(nn__rmsnorm__apply x weight out n eps)` — `out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]`, `eps` the
 * model's own (its config's `rms_norm_eps`).
 * ⛔ THE EPSILON IS INSIDE THE SQUARE ROOT, NOT ADDED TO IT, and the old tree puts it there too:
 * `rsqrt(sum/n + eps)`. Outside it would be a different function, and on a near-zero row the difference
 * is not small — it is the difference between a finite answer and a division by zero. */

/* `(nn__argmax__find x n)` — the INDEX of the largest element, as an ordinary integer.
 * ⛔ TIES GO TO THE LOWEST INDEX, which is `>` rather than `>=` in the walk and is the same rule the old
 * tree's fold uses. It is stated because a tie is not rare in a freshly-zeroed buffer, and two
 * implementations that disagree about it disagree about the TOKEN a model emits. */

/* `(nn__softmax__apply x out n)` — `out[i] = exp(x[i] - max x) / Σ_j exp(x[j] - max x)`.
 * ⛔⛔ THE MAXIMUM IS SUBTRACTED AND THAT IS NOT AN OPTIMISATION, IT IS WHAT MAKES THE FUNCTION
 * COMPUTABLE. `expf` overflows fp32 at 88 and half's range is narrower still, so the textbook form
 * answers `inf/inf` on logits a real router produces. ⛳ AND SUBTRACTING IT CHANGES NOTHING:
 * `exp(a−m)/Σexp(b−m) == exp(a)/Σexp(b)` EXACTLY, because `exp(−m)` cancels top and bottom — so the
 * stable form is the same function, not an approximation of it.
 * ⇒ ★ AND THE DIVISOR CAN NEVER BE ZERO, because the maximum's own term is `exp(0) == 1`, so the sum
 * is at least 1. That is a consequence of the subtraction and not a guard somebody has to remember.
 * ⛳ `x` MAY ALIAS `out`: every write touches the index its read just consumed. */

/* `(nn__sigmoid__apply x out n)` — `out[i] = 1 / (1 + exp(−x[i]))`.
 * ⛳ WRITTEN WITH `exp(−x)` AND NOT `exp(x)/(1+exp(x))`, for the reason `swiglu` gives beside its own
 * silu: `expf(−x)` overflows for no positive `x` and underflows harmlessly for negative ones, while
 * `expf(x)` overflows fp32 at 88. The two are the same function and only one of them is total.
 * ⛔ IT IS A VERB OF ITS OWN RATHER THAN A FOLD INSIDE SOMETHING, because `shared_expert_gate` needs
 * it ALONE — `sigmoid(gate) * shared_out` — where `swiglu` fuses its silu into a product it also
 * performs. ⇒ ★ A FUSED ACTIVATION IS NOT AVAILABLE TO A CALLER THAT WANTS ONLY THE ACTIVATION.
 * ⛳ `x` MAY ALIAS `out`, for the same reason softmax's may. */

/* ══ ⭐ THE GATE CHAIN — THREE POINTWISE VERBS DELTANET NEEDS AND NOTHING ELSE DOES YET ═════════════
 * `MEASURED` by reading the installed `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`
 * (transformers 5.13.0), not from memory of it. `Qwen3_5MoeGatedDeltaNet.forward:621` is the whole of
 * why these three exist:
 * ```
 *   beta = b.sigmoid()                                        <- ships already
 *   g    = -A_log.float().exp() * softplus(a.float() + dt_bias)
 *   decay_t = g.exp()                    (torch_recurrent_gated_delta_rule:513)
 *   query, key = l2norm(., eps=1e-6)     (:466-467, then query /= sqrt(head_dim))
 * ```
 * ⛳ SO `exp` IS WANTED TWICE ON TWO DIFFERENT THINGS — once on a per-head CONSTANT (`A_log`, which a
 * loader could fold away) and once per token on `g`. It is published as one verb because the second
 * caller is unavoidable, not because the first is.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* `(nn__vector__softplus x out n)` — `out[i] = log(1 + exp(x[i]))`.
 * ⛔⛔ WRITTEN IN THE STABLE FORM AND IT IS NOT AN OPTIMISATION — the textbook expression overflows at
 * `x = 88` in fp32, and `a + dt_bias` has no bound that keeps it below that. The identity is
 * `softplus(x) == max(x,0) + log1p(exp(-|x|))`, which is the SAME function on every input where the
 * textbook one is finite, and finite everywhere else. ⇒ ★ A FUNCTION THAT IS ONLY CORRECT ON THE INPUTS
 * SOMEBODY EXPECTED IS A GUARD, NOT A DEFINITION.
 * ⛳ `x` MAY ALIAS `out`. */

/* `(nn__vector__exp x out n)` — `out[i] = exp(x[i])`.
 * ⛔ AND IT CAN OVERFLOW, WHICH IS THE CALLER'S BUSINESS AND NOT THIS VERB'S TO HIDE. `expf` passes fp32's
 * range at 88 and half's at 11.09, so a positive input of 12 answers `inf` in a buffer. The verb REFUSES
 * on overflow rather than saturating, because a saturated `exp` reads as a very large decay and the
 * recurrence it feeds would keep running with it. ⇒ ★ A VERB THAT CANNOT BE TOTAL SHOULD FAIL LOUDLY
 * RATHER THAN CHOOSE A PLAUSIBLE NUMBER.
 * ⛳ DELTANET'S OWN USE CANNOT REACH THAT: `g` is `-exp(A_log) * softplus(...)`, a product of a positive
 * and a non-negative, NEGATED — so `g <= 0` and `exp(g)` lands in `(0, 1]`. That is a property of the
 * caller, stated here so nobody reads the refusal as unreachable code.
 * ⛳ `x` MAY ALIAS `out`. */

/* `(nn__vector__l2norm x out n)` — `out[i] = x[i] * rsqrt(Σ_j x[j]^2 + 1e-6)`.
 * ⛔⛔ IT IS A **SUM**, NOT A MEAN, AND THAT IS THE ONE THING TO GET RIGHT HERE. `nn__rmsnorm__apply`
 * three declarations up divides by `n` and this does not, so on a 128-wide head the two differ by a
 * factor of `sqrt(128) = 11.3`. HF's own source carries a comment about exactly this divergence
 * (`modeling_qwen3_5_moe.py:294`, on `l2norm` vs `GatedRMSNorm`), which is evidence it catches people.
 * ⇒ ★ TWO NORMALISERS IN ONE MODEL THAT DIFFER ONLY IN A DIVISION IS A TRAP, SO THE VERBS ARE NAMED
 * APART AND SAY SO IN EACH OTHER'S PROSE.
 * ⛳ THE EPSILON IS INSIDE, as it is for `rmsnorm`, and for the same reason: outside it is a different
 * function, and on a zero row the difference is finite-versus-divide-by-zero.
 * ⛔ THE `1/sqrt(head_dim)` SCALE ON THE QUERY IS **NOT** HERE. `torch_recurrent_gated_delta_rule:470`
 * applies it AFTER, to the query only, and folding it in would silently scale the key too.
 * ⛳ `x` MAY ALIAS `out`: the sum is complete before the first write. */

/* ══ ⭐⭐ READING A BUFFER BY INDEX — THE TWO VERBS `top_k`'s ANSWER NEEDS ═══════════════════════════
 * ⚖ RULED. ARCHITECT: *"it should return the list of ints as nodes for the lisp … and then
 * you can have a nn opcode that (nn__vector__get_values_at vector (positions)) and that returns a list
 * with the values in order of the position of the list. your at (buf,i) also works, we can have both."*
 * ⛳ WHY EITHER IS NEEDED AT ALL: `top_k` answers NAMES, and every caller that wants the VALUES at those
 * names had no way to ask. The router is the case — `softmax -> topk -> renormalise` needs the
 * probabilities of the eight it chose, and nothing in this package read one element of a buffer.
 * ⇒ ★ A VERB THAT ANSWERS INDICES IS HALF AN ANSWER UNTIL SOMETHING CAN SPEND THEM.
 * ⛔⛔ AND THE ANSWER'S TYPE IS A RULING, NOT A CONVENIENCE: ⚖ *"the return values will be the value type
 * of the vector."* A buffer holds fp16, so these answer `sys__value_float` — and the widening is EXACT,
 * because every fp16 is representable in fp64. ⛳ THAT IS WHY THE FLOAT KIND HAD TO EXIST FIRST: without
 * it the only way back was an integer, and an integer is the wrong answer for a probability.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* `(nn__vector__at x i)` — the one element at `i`, as a float.
 * ⛔ THE INDEX IS CHECKED AGAINST THE BUFFER'S OWN ROOM and refuses past it. `room()` is the CLASS the
 * allocation came from and not the payload somebody meant to put in it, so this bound is the only one
 * available here — it catches a wild index and does not catch an index inside a class but past the
 * data, which is the caller's to know. Stated because a bound that is not total reads as one that is. */

/* `(nn__vector__get_values_at x positions)` — a LIST of floats, one per position, IN THE POSITIONS' OWN
 * ORDER. ⚖ *"returns a list with the values in order of the position of the list."*
 * ⛳ SO IT IS NOT A GATHER-AND-SORT AND NOT A SET: repeats are honoured and order is the caller's.
 * ⛔ EVERY POSITION MUST BE AN INT AND IN RANGE, AND ONE BAD ONE REFUSES THE WHOLE CALL rather than
 * answering a short list. A short list would be indistinguishable from a caller that asked for fewer.
 * ⛔ AN EMPTY POSITION LIST **REFUSES**. `MEASURED` — a list with no cells does not satisfy
 * `sys__sublist__is`, so this verb cannot tell it apart from something that is not a list at all and
 * takes the safe arm. ⛳ THE SENTENCE HERE ONCE CLAIMED THE OPPOSITE, on the reasoning that *"asking for
 * nothing is a question with an answer"* — which is good reasoning about a contract and was not a claim
 * about this code. Nothing wants the capability, so the contract follows the tier's limit.
 * ⇒ ★ A CONTRACT WRITTEN BEFORE THE TEST RAN IS A DESIGN INTENTION, AND THE TWO ARE ONLY THE SAME
 * DOCUMENT AFTER SOMEBODY RUNS IT.
 * ⛔⛔ AND THE ANSWER IS TAGGED `QUOTED_LIST`, NEVER `OBJECT_REFERENCE` — ▶ the impl, where the reason is
 * written out. An ordinary reference to a list is exactly what the evaluator's scan treats as a form
 * still to reduce, so answering under that tag makes the machine try to RUN the answer. */
static __device__ __noinline__ void nn__vector__zzpackage_apply_get_values_at(sys__heap_node* base,
                                                                              uint64_t form);

/* ══ ⭐⭐⭐ A VIEW INTO A BUFFER — WHAT LETS A LAYER BE WRITTEN AS A PROGRAM ════════════════════════
 * ⚖ RULED. ARCHITECT: *"nn-12 looks like it needs nn__vector__range(vector, start, end) to
 * return a vector view, either as a copy to a destination or as a reference."*
 *
 * ⛔⛔ THE GAP IT CLOSES, `MEASURED` BY ENUMERATION: `in_proj_qkv` answers ONE 8192-wide buffer that a
 * DeltaNet layer must cut into q/k/v and then into 32 heads of 128, and nothing could offset into a
 * buffer. `nn__expert__plane` mints a view from a COLLECTION slot; an activation lives in the pool, and
 * the pool had no equivalent. Every cut in `src_deltanet_layer_oracle.py` was therefore done on the
 * HOST, which is why that file proved the arithmetic and not the program.
 *
 * ⭐⭐ AND IT NEEDS NO NEW KIND, WHICH IS THE PART WORTH READING TWICE. `nn__weights` is already *"an
 * address and a length, and it owns NEITHER"*, and `nn__primitives__room` accepts EXACTLY TWO kinds —
 * `nn__buffer` and `nn__weights`. So a view minted as `nn__weights` over a buffer's bytes is accepted by
 * **every verb in this package with no change to any of them**. ⇒ ★ THE RIGHT NEW VERB IS OFTEN A
 * SECOND WAY TO MINT AN OBJECT THAT ALREADY EXISTS, NOT A NEW OBJECT.
 *
 * ── ⚖ A REFERENCE AND NOT A COPY, AND THE REASON IS DIRECTION ───────────────────────────────────────
 * The head loop needs the cut BOTH WAYS: read head `h`'s 128 values out of `conved`, and write head
 * `h`'s answer back into `core`. A view does both with one verb; a copy needs two, and the second one
 * would have to know it is a scatter. ⛳ The architect offered either; this is why the reference wins.
 *
 * ── ⛔ AND THE LENGTH TRAVELS, WHICH IS WHY IT IS `range` AND NOT `start_at` ─────────────────────────
 * ⚖ *"maybe it will be nn__vector__start_at(vector, position) and return an address."* A bare address
 * would be accepted everywhere and would carry no bound — so `room()` would answer for the WHOLE
 * buffer and every verb's length check would go on passing while measuring the wrong thing. That is a
 * check that survives in form while admitting everything, which is worse than one that is absent.
 * ⇒ ★★ A BOUND THAT IS STILL COMPUTED AND MEASURES THE WRONG SPAN IS THE MOST EXPENSIVE KIND TO OWN,
 * BECAUSE NOTHING GOES RED ON THE DAY IT CEASES TO DISCRIMINATE. The count travels with the address.
 *
 * ── ⭐⭐⭐ A VIEW HOLDS WHAT IT LOOKS AT — ⚖ RULED ────────────────────────────────────────
 * ⚖ ARCHITECT: *"i think the view does a hold once on the held object, and then other references to the
 * view object are held by the view object itself … the base method should hold and it is the developer
 * responsibility to do a nohold only if it is sensible."*
 * ⛳ SO `nn__weights` CARRIES A THIRD WORD, `HELD`, AND THE KIND SERVES TWO OWNERS. A PAGE view
 * (`nn__expert__plane`) stores 0 — a page outlives every name for it and there is nobody to give a hold
 * back to. A BUFFER view stores the buffer and RETAINED it, so the pool cannot re-hand those bytes while
 * a view of them is alive. A NESTED view holds the view it came from, and the release loop drains the
 * chain, so the buffer's hold goes when the last link does.
 * ⇒ ★★ THE SAFE VERSION IS THE DEFAULT AND THE BORROWED ONE IS WHAT SOMEBODY HAS TO ASK FOR. A borrowed
 * span is a real optimisation, and it is also the kind of unsafety that produces a correct-LOOKING
 * answer out of somebody else's bytes — which is the failure this tree has the least defence against.
 * ⛳ A `nohold` VARIANT IS NOT BUILT AND IS NOT OWED: ⚖ *"at worst we implement a nohold alternative
 * where you need to pay attention not to deallocate stuff and only do it if you can borrow someone
 * else's hold."* Nothing has asked for one, and an unused unsafe verb is a trap armed for whoever
 * finds it.
 *
 * `(nn__vector__range v start count)` — `count` ELEMENTS from element `start`, as an `nn__weights`.
 * ⛳ IT COUNTS IN ELEMENTS, NOT BYTES, because every other verb in this package takes an element count
 * and a caller who has to multiply by 2 at one call site out of twenty will eventually not. */
static __device__ __noinline__ void nn__vector__zzpackage_apply_range(sys__heap_node* base,
                                                                      uint64_t form);

/* `(nn__vector__zero v n)` — `n` elements set to zero, answering `v`.
 * ⚖ RULED. ARCHITECT: *"now add the zero verb so the boot is a program too."*
 *
 * ⛔⛔ THE GAP IT CLOSES IS **THE BOOT**, NOT THE ARITHMETIC. A composed layer runs entirely on the card
 * — but a conversation's FIRST token needs a zeroed recurrent state and a zeroed conv window, and a
 * program had no way to put a value into a buffer at all: this language has no literal wider than an
 * integer, and `getnew` hands back whatever the pool last held. So the two things that must start at a
 * known value were the host's, and *"the layer is a program"* was true of the arithmetic and not of the
 * start. ⇒ ★ A FORWARD PASS THAT CANNOT INITIALISE ITSELF IS A PROGRAM WITH A HOST-SHAPED HOLE AT
 * TOKEN ZERO.
 *
 * ⭐⭐ AND ZERO IS THE ONE FILL THAT NEEDS NO WIDTH, WHICH IS WHY IT IS ITS OWN VERB AND NOT `fill(0)`.
 * All-zero bits are zero at every width: the DeltaNet state is fp32 and this verb counts in fp16
 * elements, and it zeroes that array CORRECTLY because the byte span is what matters and every byte is
 * the same. A `fill(v, value, n)` could not do that — it has to know how wide to write `value`, so it
 * would be an fp16 verb that quietly corrupted an fp32 array. ⇒ ★★ TWO OPERATIONS THAT LOOK LIKE ONE
 * WITH A PARAMETER ARE TWO WHEN ONE OF THEM IS TOTAL AND THE OTHER IS NOT.
 * ⛳ SO A CALLER ZEROING A WIDER ARRAY PASSES ITS BYTE COUNT DIVIDED BY TWO, and that is the ONLY place
 * this package asks anyone to convert a width. It is stated here because it is the one exception.
 *
 * ⛳ IT ACCEPTS A VIEW LIKE EVERY OTHER VERB, so zeroing one head's state is
 * `(nn__vector__zero (nn__vector__range s32 <off> <n>) <n>)` and needs nothing further. */

#endif /* SILVANN__PACKAGES_NN_CPU_PRIMITIVES__HEADER_CUH */
