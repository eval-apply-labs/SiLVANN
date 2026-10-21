# `ai_deepseek_4` — DeepSeek-V4.1 as verbs

The layers of DeepSeek-V4.1-Flash as verbs of the Eval-Apply GPU Interpreter. The model has four residual streams mixed
by a hyper-connection at every site, each site collapsing them by the weights its previous site computed; an attention
of 64 query heads over one 512-wide latent a position, over a sliding window of 128 rows and — past the first two
layers — the compressed latents of a source layer; an n-gram memory written into the streams at two layers; and 384
routed experts of which 6 run for a token, with one shared expert. The package depends only on `sys` and `nn`; its
arithmetic is `nn`'s doors.

The streams are floats, and so is a site's collapse until it is normed: a sink token's pass fp16's range — the
begin-of-sentence token's 307,505 after the 40 layers in the model's own code (`measurements/2026-10-21_deepseek_streams_fp32_node03.md`),
which DeepSeek's bf16 holds. Everything a sublayer reads and writes is halves, as in every other model.

The layout it is written for: the matrices every token uses on a card, or split over a tensor-parallel group of cards;
the routed experts on the CPU, in RAM or mapped from a file, a fixed share of them on the cards (`card_experts`).

A verb takes its weights and buffers as a **table**, a node array whose cells are named in
`contracts/objects/layer.cuh`. The runtime that builds the tables and the programs is `silvann_runtime/deepseek4.py`.

A **mask** — `W' = W + Σ r_i·v_iᵀ` beside wo_b and the shared expert's down, a changed model kept as the directions its
matrices moved (`python/nn_rank1_mask_from_projection.py`) — is three cells past a site's table, which the attention and
shared verbs add to the matrix's answer.

## The verbs

| verb | what it does |
|---|---|
| `ai_deepseek_4__attention` | the attention site: its mix, the collapse, the norm, q through its rank, the latent into the window's ring, the tails turned, every head over the ring (and the compressed latents) with its sink, the tails turned back, the grouped low-rank output, the streams mixed again; a source layer also compresses its group |
| `ai_deepseek_4__route` · `ai_deepseek_4__shared` · `ai_deepseek_4__close` | the MoE site on the card: the √softplus router's picks and the rotated input into a hand-off buffer, the shared expert, the routed sum added and the streams mixed again |
| `ai_deepseek_4__experts` | on a CPU worker: its picked experts, gate-and-up, the clamped swiglu, the rotation, the weighted down — nn's routed experts |
| `ai_deepseek_4__engram` | the n-gram memory: a position's looked-up rows through their map into a key a stream and a value, each stream gated by its key |

## What it does not do yet

The indexer's first level, the candidate blocks, so the context stops at 16,384 positions; a prompt as rows — it is
read a position at a time; the routed experts over two sockets; a card's tier of experts kept as a cache, where
`card_experts` is a fixed share.

## Where to start reading

- `cpu/opcodes/layer__abi.cuh` — every verb
- `contracts/objects/layer.cuh` — the tables
- `python/silvann_runtime/deepseek4.py` — the placement, the tables, the program
- `python/silvann_runtime/engram_ds4.py` — the n-gram memory's hashing and rows, on the host
