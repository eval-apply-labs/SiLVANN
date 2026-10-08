#ifndef SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MOE_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MOE_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_QWEN_3__MOE__FAULT_TABLE  0x51334D54ull   /* "Q3MT" — the plane table is not what the verb reads   */
#define AI_QWEN_3__MOE__FAULT_EXPERT 0x51334D45ull   /* "Q3ME" — a picked expert is not resident, or not typed */
#define AI_QWEN_3__MOE__FAULT_ROOM   0x51334D52ull   /* "Q3MR" — the scratch is too small for the shapes       */

/* ⭐ THE PLANE TABLE — a node array the boot program fills once a layer (⚖ *"plane table"*), each element
 * at a fixed place. Planes are nn's weight views; the integers are this layer's shape and where its routed
 * experts live. The scratch is one card buffer the verb cuts into its intermediates; one can serve every
 * layer, since a layer's MoE finishes before the next begins. */
#define AI_QWEN_3__MOE__NORM           0u    /* post-attention RMSNorm weight, its `1 + w` already applied */
#define AI_QWEN_3__MOE__ROUTER         1u    /* the router's codes and scales (lossless: d 16)              */
#define AI_QWEN_3__MOE__ROUTER_LUT     2u
#define AI_QWEN_3__MOE__SHARED_GATE    3u    /* the shared expert: gate, up, down, and its gate logit       */
#define AI_QWEN_3__MOE__SHARED_GATE_LUT 4u
#define AI_QWEN_3__MOE__SHARED_UP      5u
#define AI_QWEN_3__MOE__SHARED_UP_LUT  6u
#define AI_QWEN_3__MOE__SHARED_DOWN    7u
#define AI_QWEN_3__MOE__SHARED_DOWN_LUT 8u
#define AI_QWEN_3__MOE__SHARED_LOGIT   9u
#define AI_QWEN_3__MOE__SHARED_LOGIT_LUT 10u
#define AI_QWEN_3__MOE__SIGNS          11u   /* the rotation's sign vector                                  */
#define AI_QWEN_3__MOE__SCRATCH        12u   /* one card buffer, cut by the verb                            */
#define AI_QWEN_3__MOE__LAYER          13u   /* integers from here on                                       */
#define AI_QWEN_3__MOE__EXPERT_TYPE    14u   /* the routed experts' collection                              */
#define AI_QWEN_3__MOE__EXPERT_UP_LUT  15u   /* offsets in a routed expert's slot: its gate_up scales,      */
#define AI_QWEN_3__MOE__EXPERT_DOWN    16u   /*   its down codes and its down scales (gate_up codes at 0)  */
#define AI_QWEN_3__MOE__EXPERT_DOWN_LUT 17u
#define AI_QWEN_3__MOE__HIDDEN         18u
#define AI_QWEN_3__MOE__INTER          19u   /* a routed expert's and the shared expert's intermediate width */
#define AI_QWEN_3__MOE__EXPERTS        20u
#define AI_QWEN_3__MOE__TOP_K          21u
#define AI_QWEN_3__MOE__ROUTER_D       22u   /* widths: the router's, the shared expert's, a routed one's   */
#define AI_QWEN_3__MOE__SHARED_D       23u
#define AI_QWEN_3__MOE__EXPERT_D       24u
#define AI_QWEN_3__MOE__TABLE          25u   /* the table's length                                          */
/* ⭐ AND FOR A PROMPT'S ROWS (`ai_qwen_3__moe_rows`), where its routed experts come from — ▶ `prefill__abi.cuh`:
 * the integer 0 in STORE for the collection; or the layer's experts in RAM, a card STAGING buffer of two halves
 * of `NN__EXPERT__GROUPS_MAX` slots, and one expert's SLOT_BYTES. The one-position verbs read no further. */
#define AI_QWEN_3__MOE__STORE          25u
#define AI_QWEN_3__MOE__STAGING        26u
#define AI_QWEN_3__MOE__SLOT_BYTES     27u
#define AI_QWEN_3__MOE__ROWS_TABLE     28u
/* ⭐ THE RESIDUAL ROTATED, for the fused verb and the rows' alike — past the streaming cells, which a table carrying it
 * fills (with zeros when it does not stream). ▶ mixer.cuh */
#define AI_QWEN_3__MOE__RESIDUAL_ROTATED 28u

/* ── ⭐ THE MoE SPLIT AT THE DEVICE SEAMS — three verbs, each its own table (⚖ *"yes that cut, each with its own
 * plane table"*). What crosses between them is small: the HAND-OFF buffer (the rotated normed input, the
 * weights, the shared expert's rotated activation), the picks (a node array the host reads), and the routed
 * sum. `ai_qwen_3__moe` stays as the three fused on one card. ───────────────────────────────────────── */

/* The hand-off buffer, in halves: [ hm rotated · H ][ weights · 16 ][ the shared activation, rotated · I ]. The
 * weights are the router's K and, at K, the shared expert's sigmoid gate. */
#define AI_QWEN_3__HAND__WEIGHTS       16u
/* The most experts a position picks — the router's `k`: the 35B and the 122B pick eight, the 397B ten. Under nn's own
 * top-k bound, which a sampler's longer lists set. */
#define AI_QWEN_3__TOP_K_MAX           16u
#define AI_QWEN_3__HAND__HALVES(H, I)  ((H) + AI_QWEN_3__HAND__WEIGHTS + (I))
/* ⭐ A PROMPT'S HAND ROWS — a row a position, each the hand above and then its `k` picks as words, so the CPU is handed
 * a chunk's rows in one copy (`pre_expert_rows` · `experts_rows` · `post_expert_rows`). */
#define AI_QWEN_3__HAND__PICKS(H, I)   ((2ull * AI_QWEN_3__HAND__HALVES(H, I) + 7ull) & ~7ull)
#define AI_QWEN_3__HAND__ROW(H, I, K)  (AI_QWEN_3__HAND__PICKS(H, I) + 8ull * (K))
/* a prompt's experts, where the slots are a cache: this many a batch, the next batch read while one computes */
#define AI_QWEN_3__EXP__BATCH          16u

/* `(ai_qwen_3__pre_expert h1 planes hand picks)` — the structure card's half before the experts */
#define AI_QWEN_3__PRE__NORM             0u
#define AI_QWEN_3__PRE__ROUTER           1u
#define AI_QWEN_3__PRE__ROUTER_LUT       2u
#define AI_QWEN_3__PRE__SHARED_GATE      3u
#define AI_QWEN_3__PRE__SHARED_GATE_LUT  4u
#define AI_QWEN_3__PRE__SHARED_UP        5u
#define AI_QWEN_3__PRE__SHARED_UP_LUT    6u
#define AI_QWEN_3__PRE__SHARED_LOGIT     7u
#define AI_QWEN_3__PRE__SHARED_LOGIT_LUT 8u
#define AI_QWEN_3__PRE__SIGNS            9u
#define AI_QWEN_3__PRE__SCRATCH          10u
#define AI_QWEN_3__PRE__HIDDEN           11u   /* integers from here on */
#define AI_QWEN_3__PRE__INTER            12u
#define AI_QWEN_3__PRE__EXPERTS          13u
#define AI_QWEN_3__PRE__TOP_K            14u
#define AI_QWEN_3__PRE__ROUTER_D         15u
#define AI_QWEN_3__PRE__SHARED_D         16u
#define AI_QWEN_3__PRE__TABLE            17u
#define AI_QWEN_3__PRE__RESIDUAL_ROTATED 17u   /* ⭐ optional: the input normed and read unrotated — ▶ mixer.cuh */

/* `(ai_qwen_3__experts hand picks planes routed)` — the routed experts, where they live */
#define AI_QWEN_3__EXP__SIGNS            0u
#define AI_QWEN_3__EXP__SCRATCH          1u
#define AI_QWEN_3__EXP__ZERO             2u    /* H zeros: the sum's residual, so `routed` is the sum alone */
#define AI_QWEN_3__EXP__LAYER            3u    /* integers from here on */
#define AI_QWEN_3__EXP__EXPERT_TYPE      4u
#define AI_QWEN_3__EXP__UP_LUT           5u
#define AI_QWEN_3__EXP__DOWN             6u
#define AI_QWEN_3__EXP__DOWN_LUT         7u
#define AI_QWEN_3__EXP__HIDDEN           8u
#define AI_QWEN_3__EXP__INTER            9u
#define AI_QWEN_3__EXP__EXPERTS          10u
#define AI_QWEN_3__EXP__TOP_K            11u
#define AI_QWEN_3__EXP__EXPERT_D         12u
/* ⭐ AND WHERE THE ROUTED EXPERTS ARE ON DISK — nn's backing, a node array (`nn/contracts/objects/expert.cuh`),
 * for the ones not in memory (⚖ *"load the misses from disk"*); the integer 0 when every expert is resident. */
#define AI_QWEN_3__EXP__BACKING          13u
#define AI_QWEN_3__EXP__TABLE            14u
/* ⭐ AND WHICH PART OF EACH EXPERT THIS WORKER HOLDS — ⚖ *"do the tensor parallelism via the two cpu sockets"* ·
 * *"save on each ram chunk their side of the tensor parallelism experts, so that we do not need numa crossings"*.
 * Read only by `experts_up` / `experts_down`. Part `s` of `P` holds, of every expert, the gate rows s·I/P … and the
 * same up rows (a gate_up of I/P rows, gate then up) and the down rows s·H/P … (H/P rows of I) — whole rows, so no
 * split lands inside a row. The table's UP_LUT, DOWN and DOWN_LUT are that part's offsets inside its slot. */
#define AI_QWEN_3__EXP__PART             14u
#define AI_QWEN_3__EXP__PARTS            15u
#define AI_QWEN_3__EXP__PART_TABLE       16u

/* `(ai_qwen_3__post_expert h1 hand routed planes out)` — the shared down, the gate, the sums, the residual */
#define AI_QWEN_3__POST__SHARED_DOWN     0u
#define AI_QWEN_3__POST__SHARED_DOWN_LUT 1u
#define AI_QWEN_3__POST__HIDDEN          2u    /* integers from here on */
#define AI_QWEN_3__POST__INTER           3u
#define AI_QWEN_3__POST__TOP_K           4u
#define AI_QWEN_3__POST__SHARED_D        5u
#define AI_QWEN_3__POST__TABLE           6u

/* ⭐ A RANK-1 MASK ON THE DOWN PROJECTIONS — ▶ mixer.cuh. The routed experts share one `r` and keep a `v` an expert
 * (`V`, a row an expert); the shared expert has its own. Each is three optional cells — `r`, `v` or `V`, and a scratch
 * for the dots and the picks' indices — past the longest table, padded with zeros up to them. */
#define AI_QWEN_3__MOE__MASK           29u   /* the routed experts' three, then the shared expert's */
#define AI_QWEN_3__MOE__SHARED_MASK    32u
#define AI_QWEN_3__EXP__MASK           16u   /* past the parts' cells */
#define AI_QWEN_3__POST__MASK          6u    /* the shared expert's */
/* ⭐ optional, past the mask's cells: 1 where the routed experts' products are in integers — nn's int8 gemvs, the
 * activation quantised in blocks of 32 (`turboquant_gemv_groups_int8` / `…_sum_int8`) — for a position's experts on the
 * CPU; 0 or absent, exact. A prompt's rows on the CPU are in integers either way. */
#define AI_QWEN_3__EXP__INT8           19u


/* ⭐ THE MOST ROWS A PROMPT VERB TAKES AT ONCE (`ai_qwen_3__*_rows`): its MoE lists a pair for every pick of every
 * row on the host, `rows · TOP_K` of them. A longer prompt is several calls, each its next rows. */
#define AI_QWEN_3__ROWS_MAX            256u

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_OBJECTS_MOE_CUH */
