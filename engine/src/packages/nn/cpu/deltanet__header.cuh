#ifndef SILVANN__PACKAGES_NN_CPU_DELTANET__HEADER_CUH
#define SILVANN__PACKAGES_NN_CPU_DELTANET__HEADER_CUH
#include "../contracts/objects/deltanet.cuh" /* its constants and fault words */
/* ══ ⭐⭐⭐ DELTANET — THE RECURRENCE, AND THE TWO VERBS THE LANGUAGE CANNOT COMPOSE ═════════════════
 *
 * ⚖⚖ THE GRANULARITY, RULED. ARCHITECT: *"i think b is reasonable and if we see the lisp
 * being slow we can make a c wrapper that does c over the b verb"* — so the HEAD LOOP AND THE FIVE
 * STEPS LIVE IN THE PROGRAM, and only what the language cannot express comes down here.
 *
 * ── THE STEP, TRANSCRIBED FROM `torch_recurrent_gated_delta_rule` (`:505-516`) ──────────────────────
 * ```
 *   S is [k_dim x v_dim], per head, PERSISTENT across tokens
 *   ①  S   <- S * g              decay; `g` is one scalar per head per token
 *   ②  kv  <- k^T S              what the state predicts for this key      [v_dim]
 *   ③  d   <- (v - kv) * beta    the error, scaled — the "delta rule"      [v_dim]
 *   ④  S   <- S + k (x) d        the rank-1 update
 *   ⑤  out <- q^T S              the readout                              [v_dim]
 * ```
 * ⭐ ② AND ⑤ ARE THE SAME OPERATION — both contract over the KEY axis, which is the matrix's ROWS —
 * so they are one verb called twice, not two. ⛳ ③ IS ALREADY EXPRESSIBLE: `scale(kv, -1)`, `add`,
 * `scale(beta)`, all shipping. ⛔ ① AND ④ ARE NOT, and neither is ②/⑤ at this width.
 *
 * ── ⛔⛔ WHY THESE TWO EXIST AT ALL, WHICH IS A STATEMENT ABOUT THE WIDTH ────────────────────────────
 * ⚖ THE STATE IS fp32 — RULED, and the price was quoted: 32 heads x 128 x 128 x 4 B is
 * **2.00 MiB a layer, 60 MiB for all thirty**. Every other buffer in this package is fp16, so
 * `nn__primitives__fits` is fp16's bound and these verbs carry their own. ⇒ ★ THE STATE IS THE FIRST
 * fp32 ARRAY IN `nn`, AND THAT IS THE WHOLE REASON `matvec_transposed` COULD NOT BE REUSED FOR ⑤ —
 * it moves halfwords, and a halfword is not an element of this matrix.
 * ⛳ WHY fp32 AND NOT fp16: the decay is a REPEATED MULTIPLY into the same array, which is the one
 * place rounding compounds instead of cancelling. ⛔ AND THE MEASUREMENT BEHIND IT IS NOT ABOUT THE
 * STATE BUT ABOUT `g`: `A` is initialised uniform on [0.01, 16], so `g = -exp(A_log)*softplus(...)`
 * reaches -27.6 and `exp(g)` is 1e-12 — four orders below half's smallest subnormal, 5.96e-08.
 * `MEASURED`: **5 of 32 heads' decay stores as exactly 0.0 in fp16.**
 * ⇒ ⭐⭐ WHICH IS WHY `g` IS A **FLOAT SCALAR** AND NOT AN ELEMENT OF A BUFFER. ⚖ The architect's
 * sketch was `(nn__deltanet__rank_1_update S k d)`; `g` had to join it, because the decay is DATA and
 * the fusion it enables is what makes ① and ④ one pass. Passing it as a `sys__value_float` — the way
 * `nn__vector__scale` takes its factor — designs the underflow out instead of bounding it.
 * ⛔⛔ AND THE ARGUMENT IS THE **LOG** DECAY, `g` ITSELF, NOT `exp(g)` — ⚖ RULED, and this is
 * the half that makes the rest work. A scalar float could carry `exp(g)` safely, but a PROGRAM has to
 * GET it: the only route from a buffer to a scalar is `nn__vector__at`, and computing `exp(g)` into a
 * half buffer first stores **zero** for every head below 5.96e-08. `g` is -91.58 and half holds it
 * exactly. ⇒ ★★ IT IS NOT ENOUGH FOR A VALUE TO BE REPRESENTABLE WHERE IT IS SPENT; IT HAS TO BE
 * REPRESENTABLE EVERYWHERE IT TRAVELS. The exponential is taken inside, in float, beside the multiply.
 *
 * ── ⛔ AND THE DIMENSIONS ARE ARGUMENTS, WHICH IS NOT A CHOICE ───────────────────────────────────────
 * `nn__primitives__room` answers the CLASS an allocation came from, never the payload somebody put in
 * it — so a verb cannot learn `k_dim` or `v_dim` by asking the buffer. Every other verb in this package
 * takes its `n` for the same reason. ⇒ ★ A SHAPE THAT IS NOT WRITTEN DOWN IS NOT AVAILABLE TO BE READ.
 *
 * ── ⭐ WHERE `S` LIVES — ⚖ *"S has the same shape of an expert, so my feel is it should have a page
 * like them"* ──────────────────────────────────────────────────────────────────────────────────────
 * A collection slot, reached exactly as an expert's bytes are, through the three-deep index as
 * `[layer][type][head]`. ⛳ THESE VERBS TAKE NO POSITION ON THAT: `nn__primitives__room` answers for a
 * PLANE and for a BUFFER alike, so the state can arrive either way and the arithmetic does not change.
 * ⚠ The COLLECTION is what must be `anchored` — a state slot is the only copy of something no re-read
 * can recover. ▶ `expert__header`'s polarity note.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* `(nn__deltanet__rank_1_update S k d g k_dim v_dim)` — steps ① and ④ in ONE pass:
 * ```
 *   for i in 0..k_dim-1:   for j in 0..v_dim-1:   S[i][j] <- g*S[i][j] + k[i]*d[j]
 * ```
 * ⭐⭐ THE DECAY IS FUSED IN AND THAT IS THE ONLY OPTIMISATION HERE THAT MATTERS. ⚖ *"that optimization
 * lives inside in the c code."* The recurrence is MEMORY-bound, not compute-bound — 2 MiB read and
 * written per layer, 120 MiB a token across thirty — so folding a separate scale pass into the update
 * turns three walks of the state into one. The arithmetic is ~49K FLOP a head, which is nothing.
 * ⛔ `S` IS fp32, `k` AND `d` ARE fp16. The state is the only wide array; what it is updated FROM comes
 * out of ordinary buffers.
 * ⛳ ANSWERS `S`'s own reference, the way every writing verb answers its output.
 *
 * ⛔⛔⛔ AND THE FUSION MOVES STEP ① EARLIER THAN THE REFERENCE PUTS IT, WHICH IS A CONTRACT ON THE
 * CALLER AND NOT AN INTERNAL DETAIL. HF decays the state and THEN reads `kv = k^T S` off the decayed
 * state (`:511-513`). Here the decay happens inside ④, so by the time this verb runs the caller has
 * already read ② off the UNDECAYED state.
 * ⛳ IT IS STILL EXACT, BECAUSE THE READOUT IS LINEAR IN `S`:  `k^T (g*S) == g * (k^T S)`.
 * **So the caller reads ② off the old state and scales the `v_dim`-long answer by `g`** — 128 multiplies
 * instead of walking 65,536 fp32 cells to decay them first. That trade is the whole point of the fusion.
 * ⛔ A CALLER WHO OMITS THAT SCALE GETS NO ERROR. The state simply remembers more than it should, drifting
 * further every token — `MEASURED` at 18x the correct answer's own error after FOUR tokens with a mild
 * decay, and it compounds. ▶ `src_deltanet_oracle.py` §⑦, whose negative control is exactly that omission.
 * ⇒ ★★ A FUSION THAT REORDERS OPERATIONS IS ONLY FREE WHERE SOMETHING IS LINEAR, AND THE CALLER IS THE
 * ONE WHO HAS TO KNOW WHICH SOMETHING. */

/* `(nn__deltanet__readout S x out k_dim v_dim)` — `out[j] = Σ_i S[i][j] * x[i]`, steps ② and ⑤.
 * ⭐ ONE VERB FOR BOTH BECAUSE THEY ARE ONE OPERATION: ② passes `k` and reads the state's prediction
 * for this key, ⑤ passes `q` and reads the answer. Nothing about the contraction differs.
 * ⛔ IT CONTRACTS OVER THE **ROWS**, WHICH IS THE TRANSPOSED DIRECTION FOR A ROW-MAJOR `[k_dim x v_dim]`
 * — `x` is indexed by the KEY axis and the answer by the VALUE axis. A reader who assumes the usual
 * `row . vector` gets a `v_dim`-long input and a `k_dim`-long output, which is the wrong shape and
 * (at 128 x 128) is NOT caught by a bound. ⇒ ★ A SQUARE MATRIX IS WHERE A TRANSPOSE ERROR STOPS BEING
 * A CRASH AND BECOMES AN ANSWER.
 * ⛳ THE ACCUMULATOR IS `float`, matching `rmsnorm`'s ruling and the reference, which reduces in fp32.
 * ⛔ `x` AND `out` MUST NOT ALIAS: every output element reads every row, so a write into `x` would be
 * read again by the next column. The verb cannot see the aliasing and does not try to.
 * ⛳ THE `1/sqrt(head_dim)` SCALE ON THE QUERY IS **NOT** HERE — `:470` applies it to `q` alone, before
 * the readout, and folding it in would silently scale ②'s key pass too. */

/* ══ ⭐⭐ THE CAUSAL CONV, AND ITS STATE ════════════════════════════════════════════════════════════
 * ⚖ RULED. ARCHITECT: *"regarding the conv state that also looks like it is a special
 * table so let's treat it like one."* ⇒ a collection slot, addressed by layer, beside the recurrent
 * state and saved with it.
 *
 * Qwen3.5 puts a DEPTHWISE CAUSAL convolution on the q/k/v projections before the recurrence.
 * Depthwise means each channel has its own 4-tap filter and no channel ever mixes with another —
 * which is why `conv1d.weight` is `[8192, 4]` and not a square matrix. `MEASURED` from the v3 bundle's
 * layer 0; `bias=False` in the module, so there is no bias here either.
 * ```
 *   out[c] = w[c][0]*s[c][0] + w[c][1]*s[c][1] + w[c][2]*s[c][2] + w[c][3]*x[c]
 *   s[c]  <- [ s[c][1], s[c][2], x[c] ]
 * ```
 * ⛔⛔ **THE WINDOW IS THREE DEEP AND THAT IS ARITHMETIC, NOT A CHOICE.** `causal_conv1d_update:262-265`
 * concatenates the state with the new column and convolves with `padding=0`, so a state of `n` columns
 * gives `n + 1 - 4 + 1` outputs and decode needs exactly one ⇒ `n = 3`. A four-deep state also works and
 * computes a second output nobody reads. ⛳ 8192 x 3 is **48 KiB a layer, 1.44 MiB for all thirty**.
 * ⛳ AND IT IS fp16 WHILE THE RECURRENT STATE IS fp32: this window is a COPY of freshly computed
 * projections, written once, read three times, discarded. Nothing is multiplied into itself, so no
 * drift can accumulate and the wider type would buy nothing. ⇒ ★ PRECISION IS A QUESTION ABOUT WHAT AN
 * ARRAY IS DONE TO, NOT ABOUT WHAT IT HOLDS.
 * ⛔ WHY IT MUST BE SAVED WITH `S`: a conversation's DeltaNet state is BOTH. Restore one without the
 * other and the first three tokens are wrong — mildly, plausibly, with no error anywhere.
 *
 * ── ⚠ THE ACTIVATION IS FUSED IN, AND IT IS `silu` BY CONFIG RATHER THAN BY NATURE ──────────────────
 * `causal_conv1d_update` takes an `activation` argument and the caller passes `config.hidden_act`,
 * which is `"silu"` for this architecture (`MEASURED`, `text_config.hidden_act`). Fusing it matches the
 * reference's own function boundary and saves a second pass over 8192 channels.
 * ⛔ `ASSUMED`: that every model reaching this verb wants `silu` there. **WHAT WOULD FALSIFY IT:** a
 * config whose `hidden_act` is anything else. **WHAT DEPENDS ON IT:** this one verb, and the remedy is
 * to split the activation out — there is no bare `silu` verb today, so that would be one more row.
 * ⛳ IT IS WRITTEN AS `x * sigmoid(x)` with `exp(-|x|)`, for the reason `sigmoid` gives beside its own.
 *
 * `(nn__deltanet__conv_step w state x out channels)` — answers `out`, and UPDATES `state` in place. */

#endif /* SILVANN__PACKAGES_NN_CPU_DELTANET__HEADER_CUH */
