#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_LAYER_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_LAYER_CUH

/* This file needs nothing: every name in it is its own. */
#define AI_GLM_5_3__LAYER__FAULT_TABLE  0x474C3554ull   /* "GL5T" — a table is not what the verb reads          */
#define AI_GLM_5_3__LAYER__FAULT_ROOM   0x474C3552ull   /* "GL5R" — a buffer is too small for the shapes        */
#define AI_GLM_5_3__LAYER__FAULT_EXPERT 0x474C3545ull   /* "GL5E" — a pick is not an expert in memory           */

/* ══ THE PLANE TABLES — a layer's planes first, then its integers, then its reals ═══════════════════════════
 * Every table a card verb reads begins with its site's hyper-connection map and norm, and ends its planes with the
 * rotation's signs and a SCRATCH buffer. The scratch's first bytes CARRY what the MLP site's three verbs hand one
 * another — the hyper-connection's weights and the shared expert's answer — so the three read one table and one
 * scratch; the rest is each verb's own, and nothing in it outlives the verb.
 *     scratch:  [ mix, CARRY_MIX bytes | shared, hidden halves | the verb's own ... ] */
#define AI_GLM_5_3__CARRY_MIX       256ull                      /* twice (2 + mult) · mult floats, mult ≤ 4 */
#define AI_GLM_5_3__CARRY(hidden)   (AI_GLM_5_3__CARRY_MIX + 2ull * (hidden))

/* the site's first planes, in every card table */
#define AI_GLM_5_3__HC_FN           0u
#define AI_GLM_5_3__HC_BASE         1u
#define AI_GLM_5_3__HC_SCALE        2u
#define AI_GLM_5_3__NORM            3u

/* `(ai_glm_5_3__kda streams table)` — Kimi Delta Attention */
#define AI_GLM_5_3__KDA__Q          4u      /* q, k, v, f_a, g_a and b read the same input: one grouped launch */
#define AI_GLM_5_3__KDA__K          6u
#define AI_GLM_5_3__KDA__V          8u
#define AI_GLM_5_3__KDA__FA         10u
#define AI_GLM_5_3__KDA__GA         12u
#define AI_GLM_5_3__KDA__B          14u
#define AI_GLM_5_3__KDA__FB         16u     /* f_b and g_b, each on its own rotated rank */
#define AI_GLM_5_3__KDA__GB         18u
#define AI_GLM_5_3__KDA__O          20u
#define AI_GLM_5_3__KDA__Q_CONV     22u
#define AI_GLM_5_3__KDA__K_CONV     23u
#define AI_GLM_5_3__KDA__V_CONV     24u
#define AI_GLM_5_3__KDA__A_LOG      25u
#define AI_GLM_5_3__KDA__DT_BIAS    26u
#define AI_GLM_5_3__KDA__O_NORM     27u
#define AI_GLM_5_3__KDA__STATE      28u     /* fp32, written in place */
#define AI_GLM_5_3__KDA__WINDOW     29u     /* q · k · v, three taps a channel, written in place */
#define AI_GLM_5_3__KDA__SIGNS      30u
#define AI_GLM_5_3__KDA__SCRATCH    31u
#define AI_GLM_5_3__KDA__PLANES     32u
#define AI_GLM_5_3__KDA__HIDDEN     32u     /* integers */
#define AI_GLM_5_3__KDA__MULT       33u
#define AI_GLM_5_3__KDA__ITERS      34u
#define AI_GLM_5_3__KDA__HEADS      35u
#define AI_GLM_5_3__KDA__HEAD_DIM   36u
#define AI_GLM_5_3__KDA__D          37u     /* the widths of the nine matrices, in their planes' order */
#define AI_GLM_5_3__KDA__EPS        46u     /* reals */
#define AI_GLM_5_3__KDA__HC_EPS     47u
#define AI_GLM_5_3__KDA__LOWER      48u
#define AI_GLM_5_3__KDA__TABLE      49u

/* `(ai_glm_5_3__mla streams table pos)` — latent attention over a cache of rotated latents */
#define AI_GLM_5_3__MLA__QA         4u      /* q_a and kv_a read the same input: one grouped launch */
#define AI_GLM_5_3__MLA__KVA        6u
#define AI_GLM_5_3__MLA__QB         8u
#define AI_GLM_5_3__MLA__O          10u
#define AI_GLM_5_3__MLA__QA_NORM    12u
#define AI_GLM_5_3__MLA__KVA_NORM   13u
#define AI_GLM_5_3__MLA__KVB        14u     /* the latent's expansion, halves: a head's key rows, then its value rows */
#define AI_GLM_5_3__MLA__CACHE      15u     /* the rotated latents, a row a position — written at `pos` */
#define AI_GLM_5_3__MLA__SCORES     16u     /* heads · positions floats */
#define AI_GLM_5_3__MLA__SIGNS      17u
#define AI_GLM_5_3__MLA__SCRATCH    18u
#define AI_GLM_5_3__MLA__PLANES     19u
#define AI_GLM_5_3__MLA__HIDDEN     19u
#define AI_GLM_5_3__MLA__MULT       20u
#define AI_GLM_5_3__MLA__ITERS      21u
#define AI_GLM_5_3__MLA__HEADS      22u
#define AI_GLM_5_3__MLA__Q_RANK     23u
#define AI_GLM_5_3__MLA__LATENT     24u
#define AI_GLM_5_3__MLA__NOPE       25u
#define AI_GLM_5_3__MLA__V_DIM      26u
#define AI_GLM_5_3__MLA__D          27u     /* q_a, kv_a, q_b, o */
#define AI_GLM_5_3__MLA__EPS        31u
#define AI_GLM_5_3__MLA__HC_EPS     32u
#define AI_GLM_5_3__MLA__TABLE      33u
/* ⭐ THE INDEXER, cells past the table — without them every position is attended, which is the model up to `topk` of
 *   them. With them a position attends the top `topk / kpool` pools of `kpool` positions and the pool still filling.
 *   Its own projections (q through the q rank, a key and a gate a position, a weight a head), its caches (the keys and
 *   the gates in a ring of `kpool` rows — read only to finish their pool — and the pooled keys, a row a pool), and three
 *   buffers: the pools' scores, the chosen positions, and their latents gathered. */
#define AI_GLM_5_3__IDX__WQB        33u     /* pairs: codes, scales */
#define AI_GLM_5_3__IDX__WK         35u
#define AI_GLM_5_3__IDX__WP         37u     /* a weight a head */
#define AI_GLM_5_3__IDX__GATE       39u     /* the pool's gate */
#define AI_GLM_5_3__IDX__KNORM_W    41u     /* the key's LayerNorm */
#define AI_GLM_5_3__IDX__KNORM_B    42u
#define AI_GLM_5_3__IDX__APE        43u     /* a place in the pool's bias to the gate, kpool rows of dim, halves */
#define AI_GLM_5_3__IDX__KEYS       44u
#define AI_GLM_5_3__IDX__GATES      45u
#define AI_GLM_5_3__IDX__POOLED     46u
#define AI_GLM_5_3__IDX__SCORES     47u     /* a float a pool */
#define AI_GLM_5_3__IDX__INDEX      48u     /* the chosen positions, u32, and their count after them */
#define AI_GLM_5_3__IDX__GATHER     49u     /* their latents, a row each */
#define AI_GLM_5_3__IDX__PLANES     50u
#define AI_GLM_5_3__IDX__HEADS      50u
#define AI_GLM_5_3__IDX__DIM        51u
#define AI_GLM_5_3__IDX__KPOOL      52u
#define AI_GLM_5_3__IDX__TOPK       53u     /* in positions */
#define AI_GLM_5_3__IDX__D          54u     /* wq_b, wk, weights, gate */
#define AI_GLM_5_3__IDX__TABLE      58u

/* `(ai_glm_5_3__mlp streams table)` — the dense MLP of the first layers */
#define AI_GLM_5_3__MLP__GATE       4u      /* gate and up read the same input: one grouped launch */
#define AI_GLM_5_3__MLP__UP         6u
#define AI_GLM_5_3__MLP__DOWN       8u
#define AI_GLM_5_3__MLP__SIGNS      10u
#define AI_GLM_5_3__MLP__SCRATCH    11u
#define AI_GLM_5_3__MLP__PLANES     12u
#define AI_GLM_5_3__MLP__HIDDEN     12u
#define AI_GLM_5_3__MLP__MULT       13u
#define AI_GLM_5_3__MLP__ITERS      14u
#define AI_GLM_5_3__MLP__INTER      15u
#define AI_GLM_5_3__MLP__D          16u     /* gate, up, down */
#define AI_GLM_5_3__MLP__EPS        19u
#define AI_GLM_5_3__MLP__HC_EPS     20u
#define AI_GLM_5_3__MLP__LIMIT      21u
#define AI_GLM_5_3__MLP__TABLE      22u

/* `(ai_glm_5_3__route streams table hand picks)` · `(ai_glm_5_3__shared hand table)` · `(ai_glm_5_3__close streams table
 * routed)` — the MoE site, one table for its three card verbs. The HAND is what crosses to the CPU: the rotated
 * input (hidden halves), then the picks' weights (top_k halves). */
#define AI_GLM_5_3__MOE__ROUTER     4u      /* lossless: halves, read against the input unrotated */
#define AI_GLM_5_3__MOE__BIAS       5u      /* the correction bias that chooses, and does not weigh */
#define AI_GLM_5_3__MOE__GATE       6u      /* the shared expert */
#define AI_GLM_5_3__MOE__UP         8u
#define AI_GLM_5_3__MOE__DOWN       10u
#define AI_GLM_5_3__MOE__SIGNS      12u
#define AI_GLM_5_3__MOE__SCRATCH    13u
#define AI_GLM_5_3__MOE__PLANES     14u
#define AI_GLM_5_3__MOE__HIDDEN     14u
#define AI_GLM_5_3__MOE__MULT       15u
#define AI_GLM_5_3__MOE__ITERS      16u
#define AI_GLM_5_3__MOE__EXPERTS    17u
#define AI_GLM_5_3__MOE__TOP_K      18u
#define AI_GLM_5_3__MOE__INTER      19u     /* the shared expert's */
#define AI_GLM_5_3__MOE__D          20u     /* gate, up, down */
#define AI_GLM_5_3__MOE__EPS        23u
#define AI_GLM_5_3__MOE__HC_EPS     24u
#define AI_GLM_5_3__MOE__SCALE      25u     /* the routed weights' factor */
#define AI_GLM_5_3__MOE__LIMIT      26u
#define AI_GLM_5_3__MOE__TABLE      27u

/* `(ai_glm_5_3__experts hand picks table routed)` — on a worker that holds routed experts: those numbered `first` up to
 * `first + count`, in its collection as `id - first`. It computes the picks it holds and leaves the others to the worker
 * that holds them, so the workers' sums add up to the layer's. An expert is one slot: its gate and up rows as one matrix
 * (codes, then scales), then its down rows. */
#define AI_GLM_5_3__EXP__SIGNS      0u
#define AI_GLM_5_3__EXP__ZERO       1u      /* hidden halves of zero, what the weighted sum starts from */
#define AI_GLM_5_3__EXP__SCRATCH    2u
#define AI_GLM_5_3__EXP__PLANES     3u
#define AI_GLM_5_3__EXP__LAYER      3u
#define AI_GLM_5_3__EXP__TYPE       4u      /* the experts' collection */
#define AI_GLM_5_3__EXP__UP_LUT     5u      /* offsets in a slot: the gate-and-up scales, the down codes, its scales */
#define AI_GLM_5_3__EXP__DOWN       6u
#define AI_GLM_5_3__EXP__DOWN_LUT   7u
#define AI_GLM_5_3__EXP__HIDDEN     8u
#define AI_GLM_5_3__EXP__INTER      9u
#define AI_GLM_5_3__EXP__EXPERTS    10u
#define AI_GLM_5_3__EXP__TOP_K      11u
#define AI_GLM_5_3__EXP__D          12u
#define AI_GLM_5_3__EXP__FIRST      13u     /* the experts this worker holds */
#define AI_GLM_5_3__EXP__COUNT      14u
#define AI_GLM_5_3__EXP__INT8       15u     /* 1: the products in integers (nn's int8 gemvs), 0: exact */
#define AI_GLM_5_3__EXP__LIMIT      16u
#define AI_GLM_5_3__EXP__TABLE      17u
/* ⭐ optional: WHERE THE EXPERTS ARE ON DISK — nn's backing (`nn/contracts/objects/expert.cuh`), or 0. With one, the
 * worker's slots are a cache in its own memory: an expert that is not in it is read from the file straight into a slot
 * (nn's loader), the least recently used one of the layer's given up for it. Without, every expert has its slot. */
#define AI_GLM_5_3__EXP__BACKING    17u
/* A prompt's experts, where the slots are a cache: this many a batch, the next batch read while one computes — so a
 * worker's cache needs room for two batches beside what it keeps. */
#define AI_GLM_5_3__EXP__BATCH      16u
/* A prompt's MLA rows attend this many at a time where they attend every position before them: the scores of a block are
 * `rows · heads · (first + rows)` floats of the rows verbs' work buffer. */
#define AI_GLM_5_3__MLA__ROWS_BLOCK 32u

#define AI_GLM_5_3__TOP_K_MAX       12u     /* nn's grouped launch takes twelve matrices */

/* ⭐ A PROMPT AS ROWS — its positions through a layer at once, each site a verb over them (`…_rows`), so a routed expert
 * is read once for every row that picked it, not once a row. A row of streams is `mult · hidden` halves.
 * A HAND ROW, what the card hands the CPU for one position: the input rotated (hidden halves), the router's weights
 * (`k` halves), and its picks as words (`k`), at the next eight bytes. A CARRY ROW is the site's carry — its maps and
 * the shared expert's answer — kept from the route to the close. */
#define AI_GLM_5_3__ROWS_MAX        4096ull
#define AI_GLM_5_3__HAND_PICKS(hidden, k)  ((2ull * (hidden) + 2ull * (k) + 7ull) & ~7ull)
#define AI_GLM_5_3__HAND(hidden, k)        (AI_GLM_5_3__HAND_PICKS(hidden, k) + 8ull * (k))

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_OBJECTS_LAYER_CUH */
