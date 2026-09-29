# `ai_glm_5_3` — GLM 5.3 Flash as verbs

The layers of GLM 5.3 Flash as verbs of the Eval-Apply GPU Interpreter. The model has four residual streams mixed
by a hyper-connection at every site, 34 layers of Kimi Delta Attention and 11 of multi-head latent attention with
a sparse indexer (DSA), and 288 routed experts of which 8 run for a token. The package depends only on `sys` and
`nn`; its arithmetic is `nn`'s doors.

The layout it is written for: the matrices every token uses on the card, the routed experts on the CPU — in RAM,
or mapped from a file on an NVMe disk so that RAM holds the experts in use — split by inner width over the
sockets. A token's MoE site is the card's router and shared expert, each socket's share of the eight picks, and
the card's sum.

A verb takes its weights and buffers as a **table**, a node array whose cells are named in
`contracts/objects/layer.cuh`. The runtime that builds the tables and the programs is
`silvann_runtime/glm5.py`.

## The verbs

| verb | what it does |
|---|---|
| `ai_glm_5_3__kda` | a KDA layer's attention site: the hyper-connection's collapse, the norm, the projections, the three short convolutions, the gated delta step, the output, the streams mixed again |
| `ai_glm_5_3__mla` | an MLA layer's site: the query absorbed into the latent cache, attention over it, the answer expanded; past 2,048 positions the indexer picks which positions are attended |
| `ai_glm_5_3__mlp` | the dense MLP site of the first layers |
| `ai_glm_5_3__route` · `ai_glm_5_3__shared` · `ai_glm_5_3__close` | the MoE site on the card: the router's picks and the input into a hand-off buffer, the shared expert, the routed sum added and the streams mixed again |
| `ai_glm_5_3__experts` | on a CPU worker: its share of the picked experts, gate-and-up, the clamped swiglu, the rotation, the weighted down |

A prompt as rows, a chunk of positions through a layer at once:

| verb | what it does |
|---|---|
| `ai_glm_5_3__kda_rows` · `ai_glm_5_3__mla_rows` · `ai_glm_5_3__mlp_rows` | the sites over the chunk, a row a position in order |
| `ai_glm_5_3__route_rows` · `ai_glm_5_3__close_rows` | the MoE site's card halves over the chunk, each row's carry kept between them |
| `ai_glm_5_3__experts_rows` | on a CPU worker, expert-major: each held expert runs once over every row that picked it, so a chunk reads an expert once |

## Where to start reading

- `cpu/opcodes/layer__abi.cuh` — every verb
- `cpu/table.cuh` — how a table is read
- `python/silvann_runtime/glm5.py` — the placement, the tables, the programs
- `python/silvann_runtime/experts_file.py` — the routed experts as a file mapped over the CPU's slots
