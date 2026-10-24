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
the routed experts on the CPU, in RAM or mapped from a file, and a tier of them on the cards — the experts picked most,
promoted from RAM as they are picked (`card_experts_gb`), kept on the card alone with `card_exclusive` — or a fixed share
(`card_experts`). A conversation reaches the model's whole length, 1,048,576 positions: past 16,384, the candidate
blocks of layer 20 choose where the later layers' indexers look.

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
| `ai_deepseek_4__attention_rows` · `…__route_rows` · `…__close_rows` · `…__experts_rows` | the same sites over a prompt's rows, a chunk at a time (`chunk`, 128 by default) — the CPU's experts over every row a pick names, those a card's tier holds on the card |
| `ai_deepseek_4__vision_embed` · `…__vision_block` · `…__vision_tail` | a picture through the vision tower: the patches embedded, a block over every patch, the last norm and the aligner, a row a 3 × 3 block of patches (`contracts/objects/vision.cuh`) |

## What it does not do yet

The routed experts over two sockets: their inner width (2304) is not a whole number of rotation blocks in two equal
parts, so each socket would need parts of its own width — they are one socket's RAM, run by the whole machine's
threads, or one socket's alone as a stage of a pipeline (`socket`).

## Where to start reading

- `cpu/opcodes/layer__abi.cuh` — every layer verb · `cpu/opcodes/vision__abi.cuh` — the tower's
- `contracts/objects/layer.cuh` · `contracts/objects/vision.cuh` — the tables
- `python/silvann_runtime/deepseek4.py` — the placement, the tables, the program
- `python/silvann_runtime/engram_ds4.py` — the n-gram memory's hashing and rows, on the host
