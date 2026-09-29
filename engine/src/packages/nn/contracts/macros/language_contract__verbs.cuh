#ifndef SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH

/* This file needs nothing: a macro body names nothing until something expands it. */
#define nn__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                       \
    /* Standing this package up. It is a published word rather than a C entry point the boot knows the */ \
    /* name of, because the boot does not call it — it builds a list of these and evaluates it, which  */ \
    /* is what lets a package decide for itself when among the others it wants to run.                */ \
    X(PKG,  0, "nn__package__init",   nn__package__zzpackage_apply_init,             NN__PACKAGE__INIT)  \
    /* ⭐ THE FIRST WORD A MODEL PROGRAM WILL ACTUALLY CALL. It takes a size and answers a buffer or an
     * error; which class that size lands in is the pool's business and never the caller's.             */ \
    X(PKG,  1, "nn__buffer__getnew",  nn__buffer__zzpackage_apply_getnew,            NN__BUFFER__GETNEW)  \
    /* ⭐ `cache[layer][token]`, SPELLED WHERE A PROGRAM READS IT. Three integers in — layer, token,     */ \
    /* plane — and a `nn__kv_ref` out. ⚖ *"shielded by an opcode so the deltanet can be masked"*: a     */ \
    /* layer with no KV cache is one arm in here rather than something every caller has to know.       */ \
    X(PKG,  2, "nn__kv__reference",   nn__cartridge__zzpackage_apply_reference,      NN__CARTRIDGE__REFERENCE)                                                                                                        \
    /* ⭐⭐ THE FIRST ARITHMETIC THIS LANGUAGE HAS EVER PUBLISHED. `(nn__swiglu__combine gate up out n)`  */ \
    /* — three buffers and a length, answering the out buffer so a program can chain it. ⚖ *"let's     */ \
    /* test it with swiglu combine."* It is the middle of the oracle: the two ends are ABI doors and    */ \
    /* this is the only part a reference implementation has to agree with.                              */ \
    X(PKG,  3, "nn__swiglu__combine", nn__swiglu__zzabi_adapter_combine,           NN__SWIGLU__COMBINE)                                             \
    /* ⚖ *"lets go ahead with the simple opcodes."* Three SHAPES, not three small things: an           */ \
    /* elementwise, a reduction that broadcasts back, and a reduction that answers a NUMBER.            */ \
    X(PKG,  4, "nn__vector__add",  nn__vector__zzabi_adapter_add,                NN__VECTOR__ADD) \
    X(PKG,  5, "nn__rmsnorm__apply",  nn__rmsnorm__zzabi_adapter,                  NN__RMSNORM__APPLY) \
    /* ⭐⭐ THE FIRST VERB THAT ANSWERS AN INTEGER RATHER THAN AN OBJECT — which is the last link of   */ \
    /* a decode loop: `poll` hands a program's integer answer straight to the host, so a token id       */ \
    /* needs no buffer, no address and no DMA.                                                          */ \
    X(PKG,  6, "nn__argmax__find",    nn__argmax__zzabi_adapter_find,              NN__ARGMAX__FIND)  \
    /* ⚖ *"do the vbr codec it should be pretty straightforward … import it as is for now."* The five  */ \
    /* planes of a VBR matrix to dense fp32. ⛔ NO SCALE AND NO ZERO-POINT — the per-row scale is       */ \
    /* multiplied into the codebook at pack time, so the decode is one indexed read.                    */ \
    X(PKG,  7, "nn__turboquant__decode",     nn__turboquant__zzabi_adapter_decode,               NN__TURBOQUANT__DECODE)      \
    /* ⭐ AND THE ONE THAT MAKES THE COMPRESSION PAY — the matrix is never expanded; each weight is   */ \
    /* decoded into a register and consumed in the same expression.                                     */ \
    X(PKG,  8, "nn__turboquant__gemv",       nn__turboquant__zzabi_adapter_gemv,                 NN__TURBOQUANT__GEMV)     \
    /* ⭐⭐ THE LOAD PROTOCOL — ⚖ *"an abi method that loads an expert is what is needed … the host asks  */ \
    /* for space, the device finds the appropriate slot and hands him a pointer and adds that pointer   */ \
    /* to the pending list, the host writes with DMA and then calls the completion."* THEY ARE VERBS    */ \
    /* AND NOT RAW DOORS because the page index lives in the device heap, so anything that walks it     */ \
    /* runs on the card — the choice was made for us rather than taken.                                 */ \
    /* ⚖ `reserve` TAKES THE TYPE and `assign` takes neither — the type comes out of the   */ \
    /* handed row, put there by the reservation. ⛳ The old argument, that a class was a CONSEQUENCE of  */ \
    /* an expert's bytes and so not the host's to supply, was right while a page held several sizes;    */ \
    /* every expert of a collection is one size now, so bytes determine nothing and the caller must say.*/ \
    X(PKG,  9, "nn__expert__reserve", nn__expert__zzpackage_apply_reserve,           NN__EXPERT__RESERVE) \
    X(PKG, 10, "nn__expert__assign",  nn__expert__zzpackage_apply_assign,            NN__EXPERT__ASSIGN) \
    X(PKG, 11, "nn__expert__revoke",  nn__expert__zzpackage_apply_revoke,            NN__EXPERT__REVOKE) \
    /* ⭐ AND THE LIST IS ONLY WORTH HAVING IF SOMEBODY CAN ASK IT — zero means everything reserved was  */ \
    /* published, which is one number standing in for a leak hunt.                                       */ \
    X(PKG, 12, "nn__expert__handed",  nn__expert__zzpackage_apply_handed,            NN__EXPERT__HANDED) \
    /* ⭐⭐ AND THE ONE THE PROTOCOL NEEDED AND DID NOT HAVE. The index is SPARSE — the boot stands the   */ \
    /* outer array from the model's layer count and the inner arrays are made ON DEMAND, and nothing     */ \
    /* made them. `assign` refused every expert on a freshly booted card. ⇒ ★ A SPARSE INDEX NEEDS A     */ \
    /* WORD THAT SAYS HOW WIDE A LAYER IS, and only the loader knows: the count is in the model's file.  */ \
    X(PKG, 13, "nn__expert__layer",   nn__expert__zzpackage_apply_layer,             NN__EXPERT__LAYER) \
    /* ⭐⭐ THE JOIN — WHAT LETS AN ARITHMETIC VERB READ A WEIGHT WITHOUT A SECOND COPY OF IT. Until    */ \
    /* this existed the load path put weights in PAGES and every compute verb took an `nn__buffer`      */ \
    /* whose bytes come from the POOL, so a real weight had to be loaded twice to be used once.         */ \
    /* ⛔ IT IS MINTED FROM THE INDEX AND NEVER FROM A RAW ADDRESS: `(nn__expert__plane layer expert    */ \
    /* offset bytes)` looks the expert UP and checks the span lies inside that expert's OWN slot, so a  */ \
    /* program can never name memory it was not given. ⇒ ★ THE BOUND IS CHECKED BY THE THING THAT       */ \
    /* KNOWS THE SLOT WIDTH — the same argument that makes `assign` take no class.                      */ \
    X(PKG, 14, "nn__expert__plane",   nn__expert__zzpackage_apply_plane,             NN__EXPERT__PLANE) \
    /* ⭐⭐ THE ROTATION — `R(v) = H·(S⊙v)`, and it is the GAUSSIAN family's PRECONDITION rather than an */ \
    /* optimisation beside it. A fixed codebook is only correct if every row is Gaussian; a row of      */ \
    /* trained weights is not, and this is what makes it one. ⚖ *"build that quant format with the      */ \
    /* global nf table, fwht row scaler and no block scaler."*                                          */ \
    /* ⛔ THE SIGN FLIP IS FUSED IN AND IS NOT OFFERED SEPARATELY: `H` on its own is also a rotation,    */ \
    /* and a bad one — it maps a Hadamard basis vector to a spike. A caller who can apply half a        */ \
    /* transform gets no error, just a worse distribution that still decodes.                           */ \
    /* ⚖ WHERE `S` COMES FROM IS NOT DECIDED — it arrives as a buffer, so this verb takes no position.  */ \
    X(PKG, 15, "nn__hadamard__rotate", nn__hadamard__zzabi_adapter_rotate,         NN__HADAMARD__ROTATE) \
    /* ⭐⭐⭐ THE MATVEC AT FULL PRECISION — ⚖ *"nn__expert__multiply_fp16 is the verb you are looking  */ \
    /* for."* `(… weights x out rows cols)`, every operand fp16, the accumulator fp32, answering the  */ \
    /* out buffer so a program can chain it. ⛳ IT TAKES NO `lut` AND NO WIDTH, because it has         */ \
    /* neither — which is the whole reason it is a verb of its own rather than the codec at `d == 16`. */ \
    /* ⚖ AND A TABLE WAS TO CHOOSE IT, AND THAT IS v2: *"nn__expert__multiply will be a lisp defun     */ \
    /* that is published by nn and then reworked by nn_tq to add its option … nn will publish a let    */ \
    /* list of key/val that contains the enum and the function that handles it so nn_tq by modifying   */ \
    /* that list can do code injection"*. ⚖ *"let's publish turboquant directly on the     */ \
    /* list and this becomes a v2 concern"*. So the codec's verbs are rows of this list like this one  */ \
    /* (`nn__turboquant__decode`, `nn__turboquant__gemv`), and a program names the verb it means. ⛳    */ \
    /* What the table would wait on when it comes back: its key exists (an expert's type,              */ \
    /* `contracts/objects/expert.cuh`) and its value does not — a type records its bytes, slots and    */ \
    /* anchoring, not whether its slots hold fp16 or a Gaussian width. ▶                               */ \
    /* `docs/next_release_backlog.md`.                                                                 */ \
    X(PKG, 16, "nn__expert__multiply_fp16", nn__expert__zzabi_adapter_multiply_fp16,                 \
      NN__EXPERT__MULTIPLY_FP16)                                                                                  \
    /* ⭐⭐ THE TWO THE ROUTER NEEDS, AND THEY ARE THE LAST TWO THAT CARRY NO DESIGN. A softmax and a  */ \
    /* sigmoid have one definition each and numpy checks them outright — which is exactly why they   */ \
    /* are here and `top-k` is not: top-k has to say WHAT SHAPE its answer takes, and that is a      */ \
    /* ruling. ⇒ ★ "UNAMBIGUOUS" IS A PROPERTY OF THE OUTPUT TYPE, NOT OF THE MATHEMATICS.           */ \
    X(PKG, 17, "nn__softmax__apply",  nn__softmax__zzabi_adapter,                 NN__SOFTMAX__APPLY) \
    /* ⛳ `shared_expert_gate` IS WHAT WANTS THIS ONE — `sigmoid(gate) * shared_out`, which `swiglu`  */ \
    /* cannot serve because it fuses its activation into a product of its own.                       */ \
    X(PKG, 18, "nn__sigmoid__apply",  nn__sigmoid__zzabi_adapter,                 NN__SIGMOID__APPLY)                                                                                  \
    /* ⭐⭐ THE ELEMENTWISE PRODUCT, AND IT HAS NO CALLER IN THE BUILT MODEL. The MoE block's        */ \
    /* `sigmoid(shared_expert_gate(h)) * shared_expert_output` is a SCALAR times a vector — that      */ \
    /* gate is `nn.Linear(hidden, 1)`, so the `*` is a broadcast and `nn__vector__scale` serves it.   */ \
    /* ⇒ ★ A PYTORCH `*` IS NOT EVIDENCE OF AN ELEMENTWISE OPERATION. ⚖ Kept and flagged.             */ \
    /* ▶ `docs_history_internal/nn_retired_symbols.md`.                                               */ \
    X(PKG, 19, "nn__vector__pointwise_mul", nn__vector__zzabi_adapter_pointwise_mul,                 \
      NN__VECTOR__POINTWISE_MUL)                                                                        \
    /* ⭐⭐ AND THE DOT PRODUCT IS A DIFFERENT VERB, NOT A SECOND NAME. The pointwise product answers */ \
    /* `n` values; this answers ONE. ⇒ ★ TWO OPERATIONS THAT DIFFER IN THE SHAPE OF THEIR ANSWER ARE */ \
    /* TWO VERBS. ⛳ It answers a `sys__value_float`, which a day ago it could not have.              */ \
    X(PKG, 23, "nn__vector__dot_product", nn__vector__zzabi_adapter_dot_product,                      \
      NN__VECTOR__DOT_PRODUCT)                                                                          \
    /* ⭐⭐⭐ ⚖ *"multiply a vector for a matrix and then multiply it for its transposed matrix."*     */ \
    /* ⭐ IT NEEDS NO TRANSPOSE: `W` is read as stored and only the ACCUMULATION changes.             */ \
    /* ⛔ fp16 ONLY. ⚖ *"we only reason in float fp16 or turboquant terms at steady compression per  */ \
    /* table."* At fp16 a halfword IS a weight; at a TurboQuant width every row is rotated along its  */ \
    /* CONTRACTION AXIS, and transposing makes the contraction run across rows — 132% wrong.         */ \
    /* `MEASURED`; ▶ the header.                                                                      */ \
    X(PKG, 24, "nn__matrix__matvec_transposed", nn__matrix__zzabi_adapter_matvec_transposed,          \
      NN__MATRIX__MATVEC_TRANSPOSED)                                                                                  \
    /* ⭐⭐ ⚖ *"nn__vector__scale is the right verb."* — RULED, and it closes `NN-8`. The  */ \
    /* routed sum is `out += w_j * y_j` with a SCALAR, which `mul` could only do by filling a whole */ \
    /* buffer with one repeated number. ⛔ THE FACTOR COMES FROM A BUFFER because this language has */ \
    /* NO FLOAT LITERAL — `int`, `true`, `false`, `null`, `error` and nothing else.                 */ \
    X(PKG, 20, "nn__vector__scale_at", nn__vector__zzabi_adapter_scale_at,         NN__VECTOR__SCALE_AT) \
    /* ⭐⭐ AND THE SCALAR FORM, WHICH ONLY BECAME EXPRESSIBLE WHEN `sys` GREW A FLOAT — ⚖ *"get the   */ \
    /* scalar version too."* The two are not redundant: this is for a factor a PROGRAM knows, `_at`   */ \
    /* for one it must READ out of a buffer something upstream computed.                              */ \
    X(PKG, 21, "nn__vector__scale",   nn__vector__zzabi_adapter_scale,             NN__VECTOR__SCALE)   \
    /* ⭐⭐ ⚖ *"a transpose method in nn where you pass a table and it gives you its transposed one."*  */ \
    /* ⛔ IT REFUSES TO WORK IN PLACE — a square matrix swaps pairs and a rectangular one needs a      */ \
    /* cycle walk, so allowing the square case would make the algorithm depend on numbers the call    */ \
    /* site does not show. ▶ the header.                                                              */ \
    X(PKG, 22, "nn__matrix__transpose", nn__matrix__zzabi_adapter_transpose,       NN__MATRIX__TRANSPOSE) \
    /* ⭐ THE GATE CHAIN — the three DeltaNet needs and nothing else does yet. ▶ `primitives__header`  */ \
    /* for the transcription of `modeling_qwen3_5_moe.py:621` these three come from.                  */ \
    X(PKG, 25, "nn__vector__softplus", nn__vector__zzabi_adapter_softplus,         NN__VECTOR__SOFTPLUS) \
    X(PKG, 26, "nn__vector__exp",      nn__vector__zzabi_adapter_exp,              NN__VECTOR__EXP)      \
    /* ⛔ A **SUM** OF SQUARES, NOT A MEAN — `nn__rmsnorm__apply` divides by `n` and this does not,    */ \
    /* which on a 128-wide head is a factor of 11.3. HF's own source carries a comment about exactly  */ \
    /* this divergence, which is evidence it catches people.                                          */ \
    X(PKG, 27, "nn__vector__l2norm",   nn__vector__zzabi_adapter_l2norm,           NN__VECTOR__L2NORM)   \
    /* ⭐⭐ ⚖ *"your at (buf,i) also works, we can have both."* — RULED alongside `top_k`'s  */ \
    /* answer. A verb that hands back INDICES is half an answer until something can spend them, and   */ \
    /* nothing in this package read one element of a buffer. ⚖ *"the return values will be the value  */ \
    /* type of the vector"* ⇒ both answer floats, and the fp16 widening is EXACT.                     */ \
    X(PKG, 28, "nn__vector__at",       nn__vector__zzabi_adapter_at,               NN__VECTOR__AT)       \
    X(PKG, 29, "nn__vector__get_values_at", nn__vector__zzpackage_apply_get_values_at,                     \
                                                                          NN__VECTOR__GET_VALUES_AT)       \
    /* ⭐⭐⭐ DELTANET — ⚖ *"i think b is reasonable"*, so the head loop and the five steps live in the */ \
    /* PROGRAM and only what the language cannot express is here. ② and ⑤ are ONE operation called    */ \
    /* twice; ③ is already `scale`+`add`+`scale`; ① and ④ are one fused pass over the state.          */ \
    /* ⛔ THE STATE IS fp32 — the first wide array in `nn` — which is why `matvec_transposed` could    */ \
    /* not serve ⑤: it moves halfwords, and a halfword is not an element of this matrix.              */ \
    X(PKG, 30, "nn__deltanet__rank_1_update", nn__deltanet__zzabi_adapter_rank_1_update,                 \
                                                                          NN__DELTANET__RANK_1_UPDATE)     \
    X(PKG, 31, "nn__deltanet__readout", nn__deltanet__zzabi_adapter_readout,                             \
                                                                          NN__DELTANET__READOUT)           \
    /* ⭐ THE CAUSAL CONV AND ITS ROLLING WINDOW — ⚖ *"the conv state also looks like it is a special */ \
    /* table so let's treat it like one."* Depthwise, 4 taps, 3 columns retained, `silu` fused in     */ \
    /* because the reference's own function takes the activation as an argument and the caller passes */ \
    /* `config.hidden_act`. ⛔ `F.conv1d` IS A CROSS-CORRELATION: `w[0]` multiplies the OLDEST sample. */ \
    X(PKG, 32, "nn__deltanet__conv_step", nn__deltanet__zzabi_adapter_conv_step,                         \
                                                                          NN__DELTANET__CONV_STEP)         \
    /* ⭐⭐⭐ A VIEW INTO A BUFFER — ⚖ *"nn-12 looks like it needs nn__vector__range(vector, start,     */ \
    /* end) to return a vector view."* It is what lets a layer be written as a PROGRAM: the head loop  */ \
    /* cuts one 8192-wide buffer 32 ways and nothing could offset into one. ⛳ NO NEW KIND — it mints   */ \
    /* an `nn__weights`, which `nn__primitives__room` already accepts, so every verb takes it unchanged.*/ \
    X(PKG, 33, "nn__vector__range",   nn__vector__zzpackage_apply_range,             NN__VECTOR__RANGE)   \
    /* ⭐⭐⭐ ⚖ *"now add the zero verb so the boot is a program too."* A conversation's FIRST token    */ \
    /* needs a zeroed state and a zeroed conv window, and `getnew` hands back whatever the pool last  */ \
    /* held. ⛳ ZERO IS THE ONE FILL THAT NEEDS NO WIDTH — all-zero bits are zero at every width, so   */ \
    /* this fp16-counting verb zeroes the fp32 state correctly. `fill(value)` could not.              */ \
    X(PKG, 34, "nn__vector__zero",    nn__vector__zzabi_adapter_zero,              NN__VECTOR__ZERO)    \
    /* ⭐⭐ ⚖ RULED: a value verb answers the BUFFER it wrote its value into, and this is the */ \
    /* one word that turns that into a value — only when the program truly needs the number, as the    */ \
    /* router's expert list does once a layer. ▶ `contracts/objects/result.cuh`.                         */ \
    X(PKG, 35, "nn__buffer__read",    nn__buffer__zzabi_adapter_read,              NN__BUFFER__READ)    \
    /* ⭐ A BUFFER IN PINNED RAM THE CARD CAN WRITE — where a value verb's `out` belongs, so the read above  */ \
    /* watches it land instead of waiting on the card. ▶ `NN__BUFFER_RAM_CLASS_LIST`.                    */ \
    X(PKG, 36, "nn__buffer__getnew_ram", nn__buffer__zzpackage_apply_getnew_ram,   NN__BUFFER__GETNEW_RAM) \
    /* ⭐⭐ ONE POSITION OF A DECODE — the rotation of every head at a position, and one query's attention  */ \
    /* over the positions a cache holds, grouped-query. What the cache is, where cos and sin come from and  */ \
    /* who writes a position into the cache are PROPOSALS, for the architect: ▶ `attention__abi.cuh`.     */ \
    X(PKG, 37, "nn__rope__apply",     nn__rope__zzabi_adapter,                     NN__ROPE__APPLY)     \
    X(PKG, 38, "nn__attention__decode", nn__attention__zzabi_adapter_decode,       NN__ATTENTION__DECODE) \
    X(PKG, 39, "nn__rope__angles",    nn__rope__zzabi_adapter_angles,              NN__ROPE__ANGLES) \
    /* ⭐⭐ ⚖ *"in the future we can add residuals"* — attention over several ranges of a cache as one      */ \
    /* softmax: a residual per range, merged two at a time, finished into halves. ▶ the verbs' file.      */ \
    X(PKG, 40, "nn__attention__residual", nn__attention__zzabi_adapter_residual,   NN__ATTENTION__RESIDUAL) \
    X(PKG, 41, "nn__attention__merge",  nn__attention__zzabi_adapter_merge,        NN__ATTENTION__MERGE)  \
    X(PKG, 42, "nn__attention__finish", nn__attention__zzabi_adapter_finish,       NN__ATTENTION__FINISH) \
    /* ⭐⭐ THE ROUTER'S CHOICE — ⚖ *"we need to have a list of some sort, which can give us the top k"*: the */ \
    /* `k` largest into the value words of a node array the program made, their softmax into halves.      */ \
    X(PKG, 43, "nn__vector__top_k",   nn__vector__zzabi_adapter_top_k,             NN__VECTOR__TOP_K) \
    /* ⭐⭐ THE SAME PRODUCT IN INTEGERS — ⚖ *"two different instructions so at boot time the lisp can decide */ \
    /* how to write its defun"*: `x` and the levels in int8, ~1% of the rows' rms from the exact gemv.    */ \
    X(PKG, 44, "nn__turboquant__gemv_int8", nn__turboquant__zzabi_adapter_gemv_int8,                    \
      NN__TURBOQUANT__GEMV_INT8)                                                                           \
    /* ⭐⭐ ONE DELTANET STEP, EVERY HEAD — ⚖ *"i thought we would do a single opcode for the deltanet to     */ \
    /* reduce fragmentation"*: the head loop's eleven verbs a head as one launch, the decay read on the card. */ \
    X(PKG, 45, "nn__deltanet__step",  nn__deltanet__zzabi_adapter_step,            NN__DELTANET__STEP)  \
    /* The loader's counts: reads picks asked for, reads predictions queued, and those a pick then wanted. */ \
    X(PKG, 46, "nn__expert__misses",  nn__expert__zzabi_adapter_misses,            NN__EXPERT__MISSES)  \
    /* ⭐⭐ THE LRU'S WORDS — ⚖ *"the prefetch and this part should be all nn verbs"*: ask for a node array of   */ \
    /* experts (the missing ones' reads queued on the loader threads), the same as a prediction, and settle  */ \
    /* a layer — wait for its reads and admit them. A `backing` is where the experts are on disk.            */ \
    X(PKG, 47, "nn__expert__request", nn__expert__zzabi_adapter_request,           NN__EXPERT__REQUEST) \
    X(PKG, 48, "nn__expert__prefetch", nn__expert__zzabi_adapter_prefetch,         NN__EXPERT__PREFETCH) \
    X(PKG, 49, "nn__expert__settle",  nn__expert__zzabi_adapter_settle,            NN__EXPERT__SETTLE) \
    /* ⭐ A BUFFER FROM ANOTHER WORKER — ⚖ *"adding a buffer roster on the host so we can move around data between */ \
    /* the cards"*: `(nn__buffer__copy from worker to bytes)` runs on the worker that receives, `from` on the     */ \
    /* worker named, through RAM — after that worker's work is done.                                           */ \
    X(PKG, 50, "nn__buffer__copy",    nn__buffer__zzabi_adapter_copy,              NN__BUFFER__COPY) \
    /* ⭐⭐ THE KV CACHE IN TIERS — ⚖ *"rotated from the start, tq8 for the warm tier … and tq4 for the cold"*: the     */ \
    /* rotation a head at a time, a rotated row quantised as the codec would, and a residual over TurboQuant rows.   */ \
    X(PKG, 51, "nn__hadamard__blocks", nn__hadamard__zzabi_adapter_blocks,         NN__HADAMARD__BLOCKS) \
    X(PKG, 52, "nn__turboquant__encode", nn__turboquant__zzabi_adapter_encode,     NN__TURBOQUANT__ENCODE) \
    X(PKG, 53, "nn__attention__residual_tq", nn__attention__zzabi_adapter_residual_tq, NN__ATTENTION__RESIDUAL_TQ) \
    /* ⭐⭐ THE HYPER-CONNECTION — a residual of several streams (GLM 5.3 Flash, DeepSeek V4): collapsed into a sublayer's */ \
    /* input by a small map of the streams, and mixed again after it. ▶ `cpu/opcodes/hyper__abi.cuh`.               */ \
    X(PKG, 54, "nn__hyper__pre",      nn__hyper__zzabi_adapter_pre,                NN__HYPER__PRE) \
    X(PKG, 55, "nn__hyper__post",     nn__hyper__zzabi_adapter_post,               NN__HYPER__POST) \
    /* ⭐⭐ KIMI DELTA ATTENTION, ONE STEP (GLM 5.3 Flash's linear attention) — DeltaNet's recurrence with a decay a key  */ \
    /* channel, its forget gate and β made inside, the output gated-rms-normed with a sigmoid. ▶ `hyper__abi.cuh`.     */ \
    X(PKG, 56, "nn__kda__step",       nn__kda__zzabi_adapter_step,                 NN__KDA__STEP) \
    /* ⭐ GLM 5.3's MoE: the router choosing by σ + bias and weighing by σ alone, and the swiglu clamped at a limit.      */ \
    X(PKG, 57, "nn__vector__top_k_biased", nn__vector__zzabi_adapter_top_k_biased, NN__VECTOR__TOP_K_BIASED) \
    X(PKG, 58, "nn__swiglu__clamped", nn__swiglu__zzabi_adapter_clamped,           NN__SWIGLU__CLAMPED) \
    /* ⭐ MULTI-HEAD LATENT ATTENTION (GLM 5.3, DeepSeek): the query absorbed into the cached latent, the answer expanded  */ \
    /* by each head's value rows — so the cache holds the latent alone and `nn__attention__decode` attends over it.     */ \
    X(PKG, 59, "nn__attention__absorb", nn__attention__zzabi_adapter_absorb,       NN__ATTENTION__ABSORB) \
    X(PKG, 60, "nn__attention__expand", nn__attention__zzabi_adapter_expand,       NN__ATTENTION__EXPAND)

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
