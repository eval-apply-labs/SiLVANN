# `ai_qwen_3` — Qwen 3.5, 3.6 and 3.8 as verbs

The layers of the Qwen 3.5 architecture family — the 27B dense model and the 35B-A3B mixture of experts among
them — as verbs of the Eval-Apply GPU Interpreter. Each verb is one site of a layer; the arithmetic inside it is
`nn`'s doors, so the package names no vendor and runs wherever `nn` does.

A verb takes its weights and buffers as a **table**: a node array whose cells are the planes (codes and scales of
each matrix), the scratch it may use, and the integers of the model's shape. The tables' cells are named in
`contracts/objects/` — `mixer.cuh` for the attention sites, `moe.cuh` for the MLP sites. The runtime that builds
the tables and the programs over them is `silvann_runtime/qwen3_5.py`.

## The verbs

A position at a time (decode):

| verb | what it does |
|---|---|
| `ai_qwen_3__deltanet` | a Gated DeltaNet layer's attention site: the grouped projections, the short convolution, the recurrent step, the gated norm, the output projection |
| `ai_qwen_3__attention` | a full-attention layer's site: gated attention with partial rotary embedding over a half-precision cache |
| `ai_qwen_3__attention_tiered` | the same over the tiered cache: the first and the most recent positions in halves, older ones at TurboQuant 8 or 4 bits |
| `ai_qwen_3__mlp` | a dense model's MLP, its norm and its residual |
| `ai_qwen_3__moe` | the whole MoE site in one verb: router, the picked experts, the shared expert, the sum |
| `ai_qwen_3__pre_expert` · `ai_qwen_3__experts` · `ai_qwen_3__post_expert` | the same site cut where another worker takes over: the router and shared expert's first half on the card, the routed experts wherever they live (the CPU, for the small-card tier), the sums back on the card |
| `ai_qwen_3__experts_up` · `ai_qwen_3__experts_down` | the routed experts split by output rows over two CPU sockets |
| `ai_qwen_3__prefetch` | the reads of the experts a layer will probably pick, started early |

A prompt as rows (prefill), a chunk of positions through a layer at once:

| verb | what it does |
|---|---|
| `ai_qwen_3__deltanet_rows` | the DeltaNet site over the chunk: projections as one GEMM, the recurrence row by row |
| `ai_qwen_3__attention_rows` · `ai_qwen_3__attention_tiered_rows` | causal attention over the chunk and the positions before it; the tiered form also cools what the chunk displaces into the lower tiers |
| `ai_qwen_3__mlp_rows` | the dense MLP over the chunk |
| `ai_qwen_3__moe_rows` | the MoE site expert-major: each picked expert runs once over every row that picked it |

## Where to start reading

- `cpu/opcodes/mixer__abi.cuh` — the attention sites, a position at a time
- `cpu/opcodes/moe__abi.cuh` — the MLP sites with experts
- `cpu/opcodes/prefill__abi.cuh` — the rows forms
- `python/silvann_runtime/qwen3_5.py` — the tables, the programs, the cache tiers, the expert placement
