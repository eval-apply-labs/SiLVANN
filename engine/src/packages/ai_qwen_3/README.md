# `ai_qwen_3` — Qwen 3.5, 3.6 and 3.8 as verbs

The layers of the Qwen 3.5 architecture family — the 27B dense model and the 35B-A3B mixture of experts among
them — as verbs of the Eval-Apply Generalized Interpreter. Each verb is one site of a layer; the arithmetic inside it is
`nn`'s doors, so the package names no vendor and runs wherever `nn` does.

A verb takes its weights and buffers as a **table**: a node array whose cells are the planes (codes and scales of
each matrix), the scratch it may use, and the integers of the model's shape. The tables' cells are named in
`contracts/objects/` — `mixer.cuh` for the attention sites, `moe.cuh` for the MLP sites with experts, `vision.cuh` for
the vision tower. A dense model's MLP site is `nn`'s `nn__mlp__apply` (and `nn__mlp__rows`), its table nn's
`contracts/objects/mlp.cuh`. The runtime that builds the tables and the programs over them is
`silvann_runtime/qwen3_5.py`.

## The verbs

A position at a time (decode):

| verb | what it does |
|---|---|
| `ai_qwen_3__deltanet` | a Gated DeltaNet layer's attention site: the grouped projections, the short convolution, the recurrent step, the gated norm, the output projection |
| `ai_qwen_3__attention` | a full-attention layer's site: gated attention with partial rotary embedding over a half-precision cache |
| `ai_qwen_3__attention_tiered` | the same over the tiered cache: the first and the most recent positions in halves, older ones at TurboQuant 8 or 4 bits |
| `ai_qwen_3__moe` | the whole MoE site in one verb: router, the picked experts, the shared expert, the sum |
| `ai_qwen_3__pre_expert` · `ai_qwen_3__experts` · `ai_qwen_3__post_expert` | the same site cut where another worker takes over: the router and shared expert's first half on the card, the routed experts wherever they live (the CPU, for the small-card tier), the sums back on the card |
| `ai_qwen_3__experts_up` · `ai_qwen_3__experts_down` | the routed experts split by output rows over two CPU sockets |
| `ai_qwen_3__prefetch` | the reads of the experts a layer will probably pick, started early |

A prompt as rows (prefill), a chunk of positions through a layer at once:

| verb | what it does |
|---|---|
| `ai_qwen_3__deltanet_rows` | the DeltaNet site over the chunk: projections as one GEMM, the recurrence row by row |
| `ai_qwen_3__attention_rows` · `ai_qwen_3__attention_tiered_rows` | causal attention over the chunk and the positions before it; the tiered form also cools what the chunk displaces into the lower tiers |
| `ai_qwen_3__moe_rows` | the MoE site expert-major: each picked expert runs once over every row that picked it |
| `ai_qwen_3__pre_expert_rows` · `ai_qwen_3__experts_rows` · `ai_qwen_3__post_expert_rows` | the MoE site's three words over the chunk with the experts on the CPU: the card's route, the CPU's experts expert-major, the card's close |
| `ai_qwen_3__stream_counts` | what streaming a prompt's experts from RAM has cost: bytes, ns copying, ns waiting on the card |

A picture through the vision tower:

| verb | what it does |
|---|---|
| `ai_qwen_3__vision_embed` · `ai_qwen_3__vision_block` · `ai_qwen_3__vision_merger` | the patch embedding, a block over every patch, the merger into the text model's width |

## Why a layer is a few verbs

⚖ The architect, 2026-09-27: *"we could do a qwen package where we combine nn words into a qwen opcode, and
then at compose time we can decide what is the situation (if it is a full layer we use the single opcode, if it
is a disjointed one with kv cache on one card, structure on a second one and expert on ram we instead use three
sub opcodes)"*; then *"a single word opcode for the attention, the expert, the part before the expert and the part
after. this way we can do the disjointed setup"*. What was ruled, and is what the package is:

- **A verb is a site, and its arithmetic is nn's.** `ai_qwen_3__moe` takes `out = h1 + MoE(h1)` — the layer's
  second residual inside it — in a few launches of nn's doors where the verbs a program would write for the same
  block make sixty-two: the norm fused with the rotation, the router, its top-k, every expert's gate_up and the
  shared expert's in one grouped gemv, the swiglus in one, the rotations in one, every down in one grouped gemv
  summed onto `h1` (`cpu/opcodes/moe__abi.cuh`). The package has no kernels: its door list is empty
  (`contracts/abi/gpu.cuh`), and the fused shapes it needs — `turboquant_gemv_groups`,
  `turboquant_gemv_groups_sum`, `swiglu_pairs`, `rmsnorm_rotate` — are nn's doors, which every family answers.
  ⚖ 2026-10-07 *"fine if they call nn opcodes internally to save abi boundaries, just make sure that if they add
  opcodes [they] really should be there or if they should be on nn"*. A Qwen fact — the shared expert's sigmoid
  gate, the gated attention, partial RoPE — stays here; what another model could use is nn's.
- **No barrier inside a verb.** Where one step needs what every block of the step before wrote, the verb
  launches again; no kernel waits for the other blocks of its own launch (`scripts/claim_rules/no_rendezvous.py`).
- **A plane table a layer.** A verb takes one node array the runtime builds at load (`contracts/objects/`), not
  a dozen operands.
- **The cut is where the host already waits.** After top-k the host reads the picks — which experts, so which
  slots — and that wait is where a disjoint layer hands the activation to the experts' worker:
  `ai_qwen_3__pre_expert` on the card, `ai_qwen_3__experts` wherever the experts live, `ai_qwen_3__post_expert`
  back on the card. The family's program (`lisp/qwen3_5.lisp`) chooses the one verb or the three as it loads;
  nothing in the package decides the topology.

## Where to start reading

- `cpu/opcodes/mixer__abi.cuh` — the attention sites, a position at a time
- `cpu/opcodes/moe__abi.cuh` — the MLP sites with experts
- `cpu/opcodes/prefill__abi.cuh` — the rows forms
- `cpu/opcodes/vision__abi.cuh` — the vision tower, its tables' cells in `contracts/objects/vision.cuh`
- `python/silvann_runtime/qwen3_5.py` — the tables, the programs, the cache tiers, the expert placement
